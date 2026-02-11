"""
RAFgenAI - PDF Upload + Hybrid ICD Extraction
Regex + LLM + Validation + GEM Conversion

Priority Logic:
1. Prefer approximate = 1
2. Fallback to approximate = 0
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import datetime

from document_processing.pdf_loader import extract_text_from_pdf
from document_processing.text_cleaner import clean_text
from document_processing.chunker import chunk_text_by_tokens

from icd_mapping.icd_regex import extract_icd_codes
from icd_mapping.icd_validator import validate_icd_codes
from icd_mapping.icd_corrector import correct_invalid_code_detailed
from icd_mapping.gem_selector import select_best_icd10_from_gem

from clinical_extraction.chain import extract_icd_from_chunk


# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------

st.set_page_config(page_title="RAF-Extract", layout="wide")
st.title("📄 Clinical PDF + Hybrid ICD Extraction")

uploaded_file = st.file_uploader("Upload Clinical PDF", type=["pdf"])

if uploaded_file is not None:

    # --------------------------------------------------
    # Save Uploaded File
    # --------------------------------------------------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Extracting text from PDF..."):
        raw_text, used_ocr = extract_text_from_pdf(tmp_path)

    os.remove(tmp_path)

    if not raw_text.strip():
        st.error("No text could be extracted from the PDF.")
        st.stop()

    st.success("PDF processed successfully!")
    st.write("Extracted word count:", len(raw_text.split()))

    if used_ocr:
        st.info("OCR was used (scanned PDF detected).")
    else:
        st.info("Digital PDF detected (no OCR needed).")

    # --------------------------------------------------
    # Clean + Chunk
    # --------------------------------------------------

    cleaned_text = clean_text(raw_text)
    chunks = chunk_text_by_tokens(cleaned_text, max_tokens=200)

    # --------------------------------------------------
    # STEP 1 — Regex Extraction
    # --------------------------------------------------

    chunk_regex_icds = [extract_icd_codes(chunk) for chunk in chunks]

    # --------------------------------------------------
    # STEP 2 — LLM Semantic Extraction
    # --------------------------------------------------

    semantic_icd_list = []
    diagnosis_objects_list = []  # Store full diagnosis objects

    with st.spinner("Running LLM semantic extraction..."):
        for i, chunk in enumerate(chunks):
            
            # Rate limiting (VERY IMPORTANT)
            if i > 0 and i % 5 == 0:
                import time
                time.sleep(1)
            
            semantic_codes, diagnoses = extract_icd_from_chunk(chunk)
            semantic_icd_list.append(semantic_codes)
            diagnosis_objects_list.append(diagnoses)

    # --------------------------------------------------
    # STEP 3 — Merge Regex + LLM
    # --------------------------------------------------

    merged_icd_list = []

    for regex_codes, llm_codes in zip(chunk_regex_icds, semantic_icd_list):
        combined = list(dict.fromkeys(regex_codes + llm_codes))
        merged_icd_list.append(combined)

    # --------------------------------------------------
    # Load Master Data
    # --------------------------------------------------

    icd10_master_df = pd.read_csv("data/icd10cm_2026.csv", dtype=str)
    icd9_master_df = pd.read_excel("data/valid_icd_9_codes.xlsx", dtype=str)

    # Rename columns to match validator expectations
    icd10_master_df = icd10_master_df.rename(columns={"code": "icd_code"})
    
    icd10_master_df["icd_code"] = icd10_master_df["icd_code"].str.replace(".", "", regex=False)
    icd9_master_df["icd_code"] = icd9_master_df["icd_code"].str.replace(".", "", regex=False)

    gem_df = pd.read_csv("data/2015_I9gem.csv", dtype=str)
    gem_df["icd9_code"] = gem_df["icd9_code"].astype(str)
    gem_df["icd10_code"] = gem_df["icd10_code"].astype(str)

    # --------------------------------------------------
    # STEP 4 — Validation + Correct Invalid Semantic Codes
    # --------------------------------------------------

    icd10_list = []
    icd9_list = []
    mapped_icd10_list = []
    final_icd10_list = []
    invalid_regex_list = []
    invalid_semantic_list = []
    corrected_codes_list = []
    correction_details_list = []
    mapping_dictionary_list = []
    gem_selections_list = []  # Track LLM selections for GEM mappings

    with st.spinner("Validating codes and correcting invalid semantic codes..."):
        for i, (regex_codes, llm_codes, merged_codes, diagnoses) in enumerate(zip(
            chunk_regex_icds, semantic_icd_list, merged_icd_list, diagnosis_objects_list
        )):

            # ---- Validate ICD-10
            matched_icd10, mismatched = validate_icd_codes(merged_codes, icd10_master_df)

            # ---- Validate ICD-9
            matched_icd9, truly_invalid = validate_icd_codes(mismatched, icd9_master_df)
            
            # ---- Separate invalid codes by source (regex vs semantic)
            invalid_regex = [code for code in regex_codes if code in truly_invalid]
            invalid_semantic = [code for code in llm_codes if code in truly_invalid]

            # ---- STEP 4.5: Correct Invalid Semantic Codes using FAISS + LLM
            corrected_codes = []
            details = []
            
            # Create mapping of invalid code to its condition
            code_to_condition = {}
            for diag in diagnoses:
                if diag.icd10 in invalid_semantic:
                    code_to_condition[diag.icd10] = diag.condition
            
            # Correct each invalid semantic code
            for invalid_code in invalid_semantic:
                
                # Rate limiting
                if len(details) > 0 and len(details) % 5 == 0:
                    import time
                    time.sleep(1)
                
                # Get the specific condition for this code
                condition_text = code_to_condition.get(invalid_code, chunks[i])
                
                try:
                    result = correct_invalid_code_detailed(
                        invalid_code=invalid_code,
                        condition_text=condition_text
                    )
                    
                    if result:
                        corrected_code = result["llm2_valid_icd_code"]
                        corrected_codes.append(corrected_code)
                        details.append(result)
                        
                        # Validate the corrected code
                        validated, _ = validate_icd_codes([corrected_code], icd10_master_df)
                        if validated:
                            # Add corrected code to matched_icd10 if valid
                            matched_icd10.extend(validated)
                
                except Exception as e:
                    print(f"Error correcting {invalid_code}: {e}")
                    continue

            # ---- GEM Mapping (for ICD-9 codes) with LLM Selection
            mapped_icd10 = []
            mapping_dict = {}
            gem_selections = {}  # Track LLM selections for multiple mappings

            # Create ICD-10 description map for LLM
            icd10_desc_map = {}
            for _, row in icd10_master_df.iterrows():
                if 'long_title' in icd10_master_df.columns:
                    icd10_desc_map[row['icd_code']] = row['long_title']

            # Get ICD-9 descriptions
            icd9_desc_map = {}
            for _, row in icd9_master_df.iterrows():
                if 'icd_code' in icd9_master_df.columns and 'long_title' in icd9_master_df.columns:
                    icd9_desc_map[row['icd_code']] = row.get('long_title', 'Unknown')
            
            # Create mapping of ICD-9 codes to evidence snippets from diagnosis objects
            # This is for ICD-9 codes that were in the original extraction
            icd9_evidence_map = {}
            for diag in diagnoses:
                # Check if this diagnosis code is an ICD-9 code
                if diag.icd10 in matched_icd9:
                    icd9_evidence_map[diag.icd10] = getattr(diag, 'evidence_snippet', '')

            for icd9_code in matched_icd9:

                normalized = icd9_code.replace(".", "")

                # Priority 1 → approximate = 1
                approx_matches = gem_df[
                    (gem_df["icd9_code"] == normalized) &
                    (gem_df["approximate"] == "1")
                ]["icd10_code"].tolist()

                # Fallback → approximate = 0
                if approx_matches:
                    selected_matches = approx_matches
                else:
                    exact_matches = gem_df[
                        (gem_df["icd9_code"] == normalized) &
                        (gem_df["approximate"] == "0")
                    ]["icd10_code"].tolist()
                    selected_matches = exact_matches

                if selected_matches:
                    # Store all possible mappings
                    mapping_dict[icd9_code] = selected_matches
                    
                    # If multiple ICD-10 codes, use LLM to select the best one
                    if len(selected_matches) > 1:
                        try:
                            icd9_desc = icd9_desc_map.get(normalized, "Unknown condition")
                            evidence_snippet = icd9_evidence_map.get(icd9_code, "")
                            
                            best_code = select_best_icd10_from_gem(
                                icd9_code=icd9_code,
                                icd9_description=icd9_desc,
                                icd10_candidates=selected_matches,
                                icd10_descriptions=icd10_desc_map,
                                clinical_context=chunks[i],
                                clinical_evidence=evidence_snippet
                            )
                            
                            if best_code and best_code not in mapped_icd10:
                                mapped_icd10.append(best_code)
                                gem_selections[icd9_code] = {
                                    "candidates": selected_matches,
                                    "selected": best_code,
                                    "method": "LLM",
                                    "evidence": evidence_snippet
                                }
                        except Exception as e:
                            # Fallback: add all codes if LLM selection fails
                            for icd10_code in selected_matches:
                                if icd10_code not in mapped_icd10:
                                    mapped_icd10.append(icd10_code)
                    else:
                        # Single mapping, add directly
                        for icd10_code in selected_matches:
                            if icd10_code not in mapped_icd10:
                                mapped_icd10.append(icd10_code)
                                gem_selections[icd9_code] = {
                                    "candidates": selected_matches,
                                    "selected": icd10_code,
                                    "method": "Single"
                                }

            # ---- Combine Final ICD-10 (now includes corrected codes!)
            combined_icd10 = list(dict.fromkeys(matched_icd10 + mapped_icd10))

            icd10_list.append(matched_icd10)
            icd9_list.append(matched_icd9)
            mapped_icd10_list.append(mapped_icd10)
            final_icd10_list.append(combined_icd10)
            invalid_regex_list.append(invalid_regex)
            invalid_semantic_list.append(invalid_semantic)
            corrected_codes_list.append(corrected_codes)
            correction_details_list.append(details)
            mapping_dictionary_list.append(mapping_dict)
            gem_selections_list.append(gem_selections)

    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------

    df = pd.DataFrame({
        "Chunk Number": range(1, len(chunks) + 1),
        "200 Token Chunk": chunks,
        "regex_icd_codes": chunk_regex_icds,
        "semantic_icd_codes": semantic_icd_list,
        "merged_icd_codes": merged_icd_list,
        "validated_icd10_codes": icd10_list,
        "validated_icd9_codes": icd9_list,
        "mapped_icd10_from_icd9": mapped_icd10_list,
        "icd9_to_icd10_mapping_dict": mapping_dictionary_list,
        "gem_llm_selections": gem_selections_list,
        "final_combined_icd10_codes": final_icd10_list,
        "invalid_regex_icd_codes": invalid_regex_list,
        "invalid_semantic_icd_codes": invalid_semantic_list,
        "corrected_codes": corrected_codes_list,
        "correction_details": correction_details_list
    })

    st.markdown("---")
    st.subheader("📑 Hybrid Extraction + Validation + GEM Mapping")

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Total Chunks: {len(chunks)}")

    # --------------------------------------------------
    # Global Final Summary
    # --------------------------------------------------

    all_final_icd10 = sorted(
        set(code for sublist in final_icd10_list for code in sublist)
    )

    st.markdown("---")
    st.subheader("🔎 Final ICD-10 Codes (Hybrid System)")

    if all_final_icd10:
        st.success(", ".join(all_final_icd10))
        st.caption(f"Total Final ICD-10 Codes: {len(all_final_icd10)}")
        
        # --------------------------------------------------
        # Final ICD-10 Codes Table with Descriptions
        # --------------------------------------------------
        
        # Load original ICD-10 master with descriptions
        icd10_with_desc = pd.read_csv("data/icd10cm_2026.csv", dtype=str)
        
        # Normalize codes for matching (remove dots)
        icd10_with_desc["icd_code_normalized"] = icd10_with_desc["code"].str.replace(".", "", regex=False)
        
        # Normalize the final codes list
        final_codes_normalized = [code.replace(".", "") for code in all_final_icd10]
        
        # Create lookup dictionary using long_title
        code_desc_map = dict(zip(
            icd10_with_desc["icd_code_normalized"], 
            icd10_with_desc["long_title"]
        ))
        
        # Create billable status lookup
        code_billable_map = dict(zip(
            icd10_with_desc["icd_code_normalized"], 
            icd10_with_desc["is_billable"]
        ))
        
        # Build final table
        final_table_data = []
        for code in all_final_icd10:
            normalized = code.replace(".", "")
            description = code_desc_map.get(normalized, "Description not found")
            is_billable = code_billable_map.get(normalized, "Unknown")
            billable_status = "Yes" if is_billable == "1" else "No" if is_billable == "0" else "Unknown"
            
            final_table_data.append({
                "icd_code": code,
                "icd_description": description,
                "is_billable": billable_status
            })
        
        final_table_df = pd.DataFrame(final_table_data)
        
        # Separate billable and non-billable codes
        billable_df = final_table_df[final_table_df["is_billable"] == "Yes"]
        non_billable_df = final_table_df[final_table_df["is_billable"] == "No"]
        
        st.markdown("---")
        st.subheader("📋 Final ICD-10 Codes with Descriptions")
        st.dataframe(final_table_df, use_container_width=True, hide_index=True)
        
        # Display billable vs non-billable summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Billable Codes", len(billable_df))
        with col2:
            st.metric("Non-Billable Codes", len(non_billable_df))
        
        # Show only billable codes table
        if not billable_df.empty:
            st.markdown("---")
            st.subheader("✅ Final Billable ICD-10 Codes with Descriptions")
            st.dataframe(billable_df[["icd_code", "icd_description"]], use_container_width=True, hide_index=True)
            st.caption(f"Total Billable Codes: {len(billable_df)}")
        
        # --------------------------------------------------
        # Display Code Corrections (FAISS + LLM)
        # --------------------------------------------------
        
        st.markdown("---")
        st.subheader("🔄 Invalid Semantic Code Corrections (FAISS + LLM)")
        
        # Aggregate all correction details
        all_correction_details = []
        for details_list in correction_details_list:
            all_correction_details.extend(details_list)
        
        if all_correction_details:
            # Create detailed correction table
            correction_table_data = []
            
            for detail in all_correction_details:
                # Format top 5 similar codes as a readable string
                top_5_formatted = "\n".join([
                    f"{code}: {desc}" 
                    for code, desc in detail["top_5_similar_codes"].items()
                ])
                
                correction_table_data.append({
                    "LLM1 Invalid Code": detail["llm1_icd_code"],
                    "LLM1 Description": detail["llm1_description"],
                    "Top 5 Similar Codes (FAISS)": top_5_formatted,
                    "LLM2 Valid Code": detail["llm2_valid_icd_code"],
                    "LLM2 Valid Description": detail["llm2_valid_description"]
                })
            
            correction_df = pd.DataFrame(correction_table_data)
            
            st.dataframe(
                correction_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Top 5 Similar Codes (FAISS)": st.column_config.TextColumn(
                        width="large"
                    )
                }
            )
            st.caption(f"Total corrections: {len(all_correction_details)}")
            
            # Show success/info message
            st.info(f"ℹ️ {len(all_correction_details)} invalid semantic codes were automatically corrected using FAISS vector similarity search + Gemini LLM.")
        else:
            st.success("✓ All semantic codes were valid. No corrections needed.")
        
        # --------------------------------------------------
        # Display GEM Multi-Mapping LLM Selections
        # --------------------------------------------------
        
        st.markdown("---")
        st.subheader("🔀 ICD-9 to ICD-10 GEM Mappings (LLM-Selected)")
        
        # Aggregate all GEM selections where LLM was used
        all_gem_selections = []
        for gem_sel_dict in gem_selections_list:
            for icd9_code, selection_info in gem_sel_dict.items():
                if selection_info.get("method") == "LLM" and len(selection_info.get("candidates", [])) > 1:
                    all_gem_selections.append({
                        "icd9_code": icd9_code,
                        "candidates": selection_info["candidates"],
                        "selected": selection_info["selected"]
                    })
        
        if all_gem_selections:
            gem_table_data = []
            
            # Load ICD-10 descriptions for display
            icd10_with_desc = pd.read_csv("data/icd10cm_2026.csv", dtype=str)
            code_desc_map = {}
            for _, row in icd10_with_desc.iterrows():
                normalized = row["code"].replace(".", "")
                code_desc_map[normalized] = row.get("long_title", "")
            
            for gem_sel in all_gem_selections:
                # Format candidates
                candidates_formatted = "\n".join([
                    f"{code}: {code_desc_map.get(code, 'Unknown')}" 
                    for code in gem_sel["candidates"]
                ])
                
                selected_desc = code_desc_map.get(gem_sel["selected"], "Unknown")
                
                gem_table_data.append({
                    "ICD-9 Code": gem_sel["icd9_code"],
                    "Available ICD-10 Mappings": candidates_formatted,
                    "LLM Selected Code": gem_sel["selected"],
                    "Selected Description": selected_desc
                })
            
            gem_df_display = pd.DataFrame(gem_table_data)
            
            st.dataframe(
                gem_df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Available ICD-10 Mappings": st.column_config.TextColumn(
                        width="large"
                    )
                }
            )
            st.caption(f"Total LLM-selected mappings: {len(all_gem_selections)}")
            st.info(f"ℹ️ When a single ICD-9 code maps to multiple ICD-10 codes, the LLM selects the most appropriate one based on clinical context.")
        else:
            st.info("No ICD-9 codes required multi-mapping selection (all had single ICD-10 mappings).")
        
    else:
        st.warning("No valid ICD-10 codes identified.")

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    os.makedirs("outputs", exist_ok=True)

    base_filename = os.path.splitext(uploaded_file.name)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{base_filename}_{timestamp}_chunks.csv"

    try:
        df.to_csv(output_path, index=False)
        st.success(f"Chunk table saved to: {output_path}")
    except PermissionError:
        st.error("File is open or locked. Please close it and retry.")

    # Download button
    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Chunk Table as CSV",
        data=csv_data,
        file_name=f"{base_filename}_{timestamp}_chunks.csv",
        mime="text/csv"
    )
