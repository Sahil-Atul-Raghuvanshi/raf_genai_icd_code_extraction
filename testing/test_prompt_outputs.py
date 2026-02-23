"""
Prompt Consistency Testing Script
Tests the same clinical notes multiple times to measure output consistency
"""

import sys
import os
from pathlib import Path
import pandas as pd
from collections import Counter
import json

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from ai_icd_extraction.scripts.document_processing.pdf_loader import extract_text_from_pdf
from ai_icd_extraction.scripts.document_processing.text_cleaner import clean_text
from ai_icd_extraction.scripts.document_processing.chunker import chunk_text_by_tokens

from ai_icd_extraction.scripts.icd_mapping.icd_validator import validate_icd_codes
from ai_icd_extraction.scripts.icd_mapping.icd_corrector import correct_codes_smart
from ai_icd_extraction.scripts.icd_mapping.gem_selector import select_best_icd10_from_gem

from ai_icd_extraction.scripts.clinical_extraction.chain import extract_icd_from_chunks_batch, reconcile_diagnoses_globally


def build_icd_lookups(icd10_df, icd9_df):
    """Build lookup dictionaries for O(1) access"""
    icd10_code_to_desc = dict(zip(icd10_df['icd_code'], icd10_df['long_title']))
    icd10_code_to_billable = dict(zip(icd10_df['icd_code'], icd10_df['is_billable']))
    
    icd9_code_to_desc = {}
    if 'icd_code' in icd9_df.columns and 'long_title' in icd9_df.columns:
        icd9_code_to_desc = dict(zip(icd9_df['icd_code'], icd9_df['long_title']))
    
    return {
        'icd10_desc': icd10_code_to_desc,
        'icd10_billable': icd10_code_to_billable,
        'icd9_desc': icd9_code_to_desc
    }


def calculate_billable_ratio(icd10_df):
    """Calculate percentage of billable codes"""
    billable_count = (icd10_df['is_billable'] == '1').sum()
    total_count = len(icd10_df)
    return billable_count / total_count if total_count > 0 else 0.85


def run_7_step_pipeline(pdf_path, icd10_master_df, icd9_master_df, gem_df, icd_lookups, billable_ratio):
    """
    Run the complete 7-step ICD extraction pipeline on a single PDF
    Returns: List of final ICD-10 codes
    """
    try:
        # Step 1: Extract text from PDF
        print(f"  Processing: {Path(pdf_path).name}")
        raw_text, used_ocr = extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            print(f"  WARNING: No text extracted from {pdf_path}")
            return []
        
        # Step 2: Clean and chunk
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text_by_tokens(cleaned_text, max_tokens=200)
        
        if not chunks:
            print(f"  WARNING: No chunks created from {pdf_path}")
            return []
        
        # Step 3: LLM Semantic Extraction (BATCH PROCESSING) - PASS 1
        semantic_icd_list = []
        diagnosis_objects_list = []
        
        batch_results = extract_icd_from_chunks_batch(chunks, batch_size=5)
        
        for semantic_codes, diagnoses in batch_results:
            semantic_icd_list.append(semantic_codes)
            diagnosis_objects_list.append(diagnoses)
        
        # Step 4: Validate ICD-10 codes
        validated_icd10_list = []
        mismatched_codes_list = []
        
        for semantic_codes in semantic_icd_list:
            matched_icd10, mismatched = validate_icd_codes(semantic_codes, icd10_master_df)
            validated_icd10_list.append(matched_icd10)
            mismatched_codes_list.append(mismatched)
        
        # Step 5: ICD-9 Fallback Detection
        validated_icd9_list = []
        truly_invalid_codes_list = []
        
        for mismatched in mismatched_codes_list:
            matched_icd9, truly_invalid = validate_icd_codes(mismatched, icd9_master_df)
            validated_icd9_list.append(matched_icd9)
            truly_invalid_codes_list.append(truly_invalid)
        
        # Step 6: ICD-9 to ICD-10 GEM Mapping
        mapped_icd10_list = []
        icd10_desc_map = icd_lookups['icd10_desc']
        icd9_desc_map = icd_lookups['icd9_desc']
        
        for i, matched_icd9 in enumerate(validated_icd9_list):
            mapped_icd10 = []
            
            for icd9_code in matched_icd9:
                normalized = icd9_code.replace(".", "")
                
                # Priority 1: approximate = 1
                approx_matches = gem_df[
                    (gem_df["icd9_code"] == normalized) &
                    (gem_df["approximate"] == "1")
                ]["icd10_code"].tolist()
                
                # Fallback: approximate = 0
                if approx_matches:
                    selected_matches = approx_matches
                else:
                    exact_matches = gem_df[
                        (gem_df["icd9_code"] == normalized) &
                        (gem_df["approximate"] == "0")
                    ]["icd10_code"].tolist()
                    selected_matches = exact_matches
                
                if selected_matches:
                    if len(selected_matches) > 1:
                        try:
                            icd9_desc = icd9_desc_map.get(normalized, "Unknown condition")
                            best_code = select_best_icd10_from_gem(
                                icd9_code=icd9_code,
                                icd9_description=icd9_desc,
                                icd10_candidates=selected_matches,
                                icd10_descriptions=icd10_desc_map,
                                clinical_context=chunks[i] if i < len(chunks) else "",
                                clinical_evidence=""
                            )
                            if best_code and best_code not in mapped_icd10:
                                mapped_icd10.append(best_code)
                        except:
                            if selected_matches[0] not in mapped_icd10:
                                mapped_icd10.append(selected_matches[0])
                    else:
                        if selected_matches[0] not in mapped_icd10:
                            mapped_icd10.append(selected_matches[0])
            
            mapped_icd10_list.append(mapped_icd10)
        
        # Step 7: FAISS Similarity Rescue + LLM Correction
        corrected_icd10_list = []
        total_truly_invalid = sum(len(x) for x in truly_invalid_codes_list)
        
        if total_truly_invalid > 0:
            for i, (truly_invalid, diagnoses) in enumerate(zip(truly_invalid_codes_list, diagnosis_objects_list)):
                corrected_codes = []
                
                if truly_invalid:
                    code_to_condition = {}
                    code_to_evidence = {}
                    
                    for diag in diagnoses:
                        if diag.icd10 in truly_invalid:
                            code_to_condition[diag.icd10] = diag.condition
                            code_to_evidence[diag.icd10] = getattr(diag, 'evidence_snippet', '')
                    
                    parallel_codes = []
                    parallel_conditions = []
                    parallel_evidence = []
                    
                    for invalid_code in truly_invalid:
                        condition_text = code_to_condition.get(invalid_code, chunks[i] if i < len(chunks) else "")
                        evidence_text = code_to_evidence.get(invalid_code, chunks[i] if i < len(chunks) else "")
                        parallel_codes.append(invalid_code)
                        parallel_conditions.append(condition_text)
                        parallel_evidence.append(evidence_text)
                    
                    smart_result = correct_codes_smart(
                        invalid_codes=parallel_codes,
                        condition_texts=parallel_conditions,
                        evidence_snippets=parallel_evidence,
                        icd10_master_df=icd10_master_df,
                        max_workers=3,
                        confidence_threshold=0.0,
                        billable_ratio=billable_ratio,
                        verbose=False
                    )
                    
                    corrected_codes_dict = smart_result["corrected_codes"]
                    for original_code, corrected_code in corrected_codes_dict.items():
                        corrected_codes.append(corrected_code)
                
                corrected_icd10_list.append(corrected_codes)
        else:
            for _ in semantic_icd_list:
                corrected_icd10_list.append([])
        
        # Combine results
        combined_icd10_per_chunk = []
        for validated, mapped, corrected in zip(validated_icd10_list, mapped_icd10_list, corrected_icd10_list):
            combined = list(dict.fromkeys(validated + mapped + corrected))
            combined_icd10_per_chunk.append(combined)
        
        # Step 8: Global Reconciliation
        try:
            reconciled_icd_codes, reconciled_diagnoses = reconcile_diagnoses_globally(
                all_chunk_results=batch_results,
                chunks=chunks,
                max_retries=2
            )
            
            if reconciled_icd_codes:
                return sorted(reconciled_icd_codes)
            else:
                # Fallback
                all_codes = []
                for codes in combined_icd10_per_chunk:
                    all_codes.extend(codes)
                return sorted(list(dict.fromkeys(all_codes)))
                
        except Exception as e:
            print(f"  WARNING: Reconciliation failed: {str(e)}")
            all_codes = []
            for codes in combined_icd10_per_chunk:
                all_codes.extend(codes)
            return sorted(list(dict.fromkeys(all_codes)))
    
    except Exception as e:
        print(f"  ERROR processing {pdf_path}: {str(e)}")
        return []


def get_largest_files_from_patient(patient_dir, num_files=10):
    """
    Get the largest PDF files from a patient directory (Discharge_Summaries + radiography_reports)
    Takes 5 largest from each folder (or all if less than 5)
    Returns list of PDF file paths sorted by size (largest first)
    """
    patient_path = Path(patient_dir)
    
    # Actual folder names based on directory structure
    discharge_dir = patient_path / "Discharge_Summaries"
    radiology_dir = patient_path / "radiography_reports"
    
    selected_files = []
    
    # Get largest files from Discharge_Summaries
    if discharge_dir.exists():
        discharge_pdfs = list(discharge_dir.glob("*.pdf"))
        if discharge_pdfs:
            # Sort by file size (largest first) and take top 5
            discharge_pdfs_sorted = sorted(discharge_pdfs, key=lambda x: x.stat().st_size, reverse=True)
            num_discharge = min(5, len(discharge_pdfs_sorted))
            selected_files.extend(discharge_pdfs_sorted[:num_discharge])
            
            print(f"  Found {len(discharge_pdfs)} files in Discharge_Summaries")
            print(f"  Selected {num_discharge} largest (sizes: {', '.join([f'{p.stat().st_size/1024:.1f}KB' for p in discharge_pdfs_sorted[:num_discharge]])})")
    
    # Get largest files from radiography_reports
    if radiology_dir.exists():
        radiology_pdfs = list(radiology_dir.glob("*.pdf"))
        if radiology_pdfs:
            # Sort by file size (largest first) and take top 5
            radiology_pdfs_sorted = sorted(radiology_pdfs, key=lambda x: x.stat().st_size, reverse=True)
            num_radiology = min(5, len(radiology_pdfs_sorted))
            selected_files.extend(radiology_pdfs_sorted[:num_radiology])
            
            print(f"  Found {len(radiology_pdfs)} files in radiography_reports")
            print(f"  Selected {num_radiology} largest (sizes: {', '.join([f'{p.stat().st_size/1024:.1f}KB' for p in radiology_pdfs_sorted[:num_radiology]])})")
    
    if len(selected_files) == 0:
        print(f"  WARNING: No PDF files found in {patient_dir}")
        return []
    
    # Sort all selected files by size (largest first)
    selected_files_sorted = sorted(selected_files, key=lambda x: x.stat().st_size, reverse=True)
    
    print(f"  Total selected: {len(selected_files_sorted)} files")
    
    return selected_files_sorted


def find_common_and_different_codes(test_results):
    """
    Analyze test results to find common ICD codes and differences
    test_results: list of ICD code lists from multiple test runs
    Returns: (common_codes, different_codes_dict)
    """
    if not test_results or len(test_results) == 0:
        return [], {}
    
    # Filter out None and empty results
    valid_results = [codes for codes in test_results if codes]
    
    if not valid_results:
        return [], {}
    
    # Find codes that appear in ALL test runs
    if len(valid_results) > 0:
        common_codes = set(valid_results[0])
        for codes in valid_results[1:]:
            common_codes = common_codes.intersection(set(codes))
        common_codes = sorted(list(common_codes))
    else:
        common_codes = []
    
    # Find codes unique to each test
    different_codes = {}
    for i, codes in enumerate(valid_results, 1):
        unique_to_test = [code for code in codes if code not in common_codes]
        if unique_to_test:
            different_codes[f"test{i}"] = sorted(unique_to_test)
    
    return common_codes, different_codes


def main():
    print("=" * 80)
    print("PROMPT CONSISTENCY TESTING")
    print("=" * 80)
    
    # Define patient directories
    patient_dirs = [
        "ai_icd_extraction/data/sample_notes/13718764",
        "ai_icd_extraction/data/sample_notes/13877234",
        "ai_icd_extraction/data/sample_notes/14119818",
        "ai_icd_extraction/data/sample_notes/16950272",
        "ai_icd_extraction/data/sample_notes/18561128"
    ]
    
    # Load master data
    print("\nLoading master data...")
    icd10_master_df = pd.read_csv("ai_icd_extraction/data/icd10cm_2026.csv", dtype=str)
    icd9_master_df = pd.read_excel("ai_icd_extraction/data/valid_icd_9_codes.xlsx", dtype=str)
    gem_df = pd.read_csv("ai_icd_extraction/data/2015_I9gem.csv", dtype=str)
    
    # Normalize columns
    icd10_master_df = icd10_master_df.rename(columns={"code": "icd_code"})
    icd10_master_df["icd_code"] = icd10_master_df["icd_code"].str.replace(".", "", regex=False)
    icd9_master_df["icd_code"] = icd9_master_df["icd_code"].str.replace(".", "", regex=False)
    gem_df["icd9_code"] = gem_df["icd9_code"].astype(str)
    gem_df["icd10_code"] = gem_df["icd10_code"].astype(str)
    
    # Build lookup dictionaries
    icd_lookups = build_icd_lookups(icd10_master_df, icd9_master_df)
    billable_ratio = calculate_billable_ratio(icd10_master_df)
    
    print(f"Master data loaded successfully")
    print(f"  ICD-10 codes: {len(icd10_master_df)}")
    print(f"  ICD-9 codes: {len(icd9_master_df)}")
    print(f"  GEM mappings: {len(gem_df)}")
    print(f"  Billable ratio: {billable_ratio:.2%}")
    
    # Create output directory and CSV file
    output_dir = Path("testing/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fixed filename (no timestamp - will override previous file)
    output_path = output_dir / "prompt_consistency_test.csv"
    
    # Initialize CSV with headers
    headers = ['filename', 'patient_id', 'test1', 'test2', 'test3', 'test4', 
               'no_of_tests_similar', 'common_icd_codes_in_all', 'different_icd_codes_per_test']
    
    # Create empty CSV with headers (will override if exists)
    pd.DataFrame(columns=headers).to_csv(output_path, index=False)
    print(f"\n✅ Output file: {output_path}")
    print(f"   (Overwriting previous results, updating after each file is processed)\n")
    
    # Collect results
    results = []
    
    # Process each patient
    for patient_dir in patient_dirs:
        patient_id = Path(patient_dir).name
        print(f"\n{'=' * 80}")
        print(f"Processing Patient: {patient_id}")
        print(f"{'=' * 80}")
        
        # Get 10 largest files from this patient (5 from each folder)
        selected_files = get_largest_files_from_patient(patient_dir, num_files=10)
        
        if not selected_files:
            print(f"  No files found for patient {patient_id}, skipping...")
            continue
        
        print(f"  Selected {len(selected_files)} random files")
        
        # Process each file
        for file_idx, pdf_path in enumerate(selected_files, 1):
            print(f"\n  File {file_idx}/{len(selected_files)}: {pdf_path.name}")
            
            # Run the pipeline 4 times for consistency testing
            test_results = []
            for test_num in range(1, 5):
                print(f"    Test {test_num}/4...", end=" ")
                
                try:
                    icd_codes = run_7_step_pipeline(
                        pdf_path=str(pdf_path),
                        icd10_master_df=icd10_master_df,
                        icd9_master_df=icd9_master_df,
                        gem_df=gem_df,
                        icd_lookups=icd_lookups,
                        billable_ratio=billable_ratio
                    )
                    test_results.append(icd_codes)
                    print(f"Done ({len(icd_codes)} codes)")
                except Exception as e:
                    print(f"FAILED: {str(e)}")
                    test_results.append([])
            
            # Analyze results
            common_codes, different_codes = find_common_and_different_codes(test_results)
            
            # Count how many tests have identical results
            if all(test_results):
                test_sets = [set(codes) for codes in test_results]
                num_similar = sum(1 for s in test_sets if s == test_sets[0])
            else:
                num_similar = 0
            
            # Create result record
            # Use just the filename or convert to absolute path first
            try:
                filename_str = str(pdf_path.resolve().relative_to(project_root))
            except ValueError:
                # If relative_to fails, just use the name
                filename_str = pdf_path.name
            
            result_record = {
                'filename': filename_str,
                'patient_id': patient_id,
                'test1': ', '.join(test_results[0]) if len(test_results) > 0 and test_results[0] else '',
                'test2': ', '.join(test_results[1]) if len(test_results) > 1 and test_results[1] else '',
                'test3': ', '.join(test_results[2]) if len(test_results) > 2 and test_results[2] else '',
                'test4': ', '.join(test_results[3]) if len(test_results) > 3 and test_results[3] else '',
                'no_of_tests_similar': f"{num_similar}/4",
                'common_icd_codes_in_all': ', '.join(common_codes) if common_codes else 'None',
                'different_icd_codes_per_test': json.dumps(different_codes) if different_codes else '{}'
            }
            
            results.append(result_record)
            
            # 🔥 WRITE TO CSV IMMEDIATELY AFTER EACH FILE
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            
            # Print summary
            print(f"    Summary: {num_similar}/4 tests identical")
            print(f"    Common codes: {len(common_codes)}")
            if different_codes:
                print(f"    Variations detected: {len(different_codes)} tests had unique codes")
            print(f"    ✅ Updated CSV: {len(results)} files processed so far")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    
    if not results:
        print("\n❌ No results collected. No files were successfully processed.")
        print("\nPossible reasons:")
        print("  1. No PDF files found in patient directories")
        print("  2. All PDFs failed to process")
        print("  3. Check directory structure matches script expectations")
        return
    
    results_df = pd.DataFrame(results)
    
    print(f"\n✅ Final CSV saved to: {output_path}")
    print(f"\nTotal files tested: {len(results_df)}")
    
    # Print consistency summary
    print(f"\n{'=' * 80}")
    print("CONSISTENCY SUMMARY")
    print(f"{'=' * 80}")
    
    # Calculate consistency metrics
    fully_consistent = len(results_df[results_df['no_of_tests_similar'] == '4/4'])
    partially_consistent = len(results_df[results_df['no_of_tests_similar'].isin(['2/4', '3/4'])])
    inconsistent = len(results_df[results_df['no_of_tests_similar'] == '1/4'])
    
    print(f"Fully Consistent (4/4): {fully_consistent}/{len(results_df)} ({fully_consistent/len(results_df)*100:.1f}%)")
    print(f"Partially Consistent (2-3/4): {partially_consistent}/{len(results_df)} ({partially_consistent/len(results_df)*100:.1f}%)")
    print(f"Inconsistent (1/4): {inconsistent}/{len(results_df)} ({inconsistent/len(results_df)*100:.1f}%)")
    
    # Analyze common codes distribution
    all_common_codes = []
    for codes_str in results_df['common_icd_codes_in_all']:
        if codes_str and codes_str != 'None':
            all_common_codes.extend(codes_str.split(', '))
    
    if all_common_codes:
        code_freq = Counter(all_common_codes)
        print(f"\nMost Common ICD Codes (appearing in common codes):")
        for code, freq in code_freq.most_common(10):
            print(f"  {code}: {freq} files")
    
    print(f"\n{'=' * 80}")
    print("TESTING COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
