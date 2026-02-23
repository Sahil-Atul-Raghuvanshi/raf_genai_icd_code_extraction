"""
FastAPI Service for ICD-10 Code Extraction
Integrates with existing ICD extraction pipeline from app.py
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import sys
import pandas as pd
from typing import List, Dict
import uvicorn

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import existing extraction functions
from scripts.document_processing.pdf_loader import extract_text_from_pdf
from scripts.document_processing.text_cleaner import clean_text
from scripts.document_processing.chunker import chunk_text_by_tokens
from scripts.icd_mapping.icd_validator import validate_icd_codes
from scripts.icd_mapping.icd_corrector import correct_codes_smart
from scripts.icd_mapping.gem_selector import select_best_icd10_from_gem_detailed
from scripts.icd_mapping.icd_vector_index import load_faiss_index
from scripts.clinical_extraction.chain import extract_icd_from_chunks_batch, reconcile_diagnoses_globally
from response_builder import build_icd_code_response_with_provenance

# Initialize FastAPI app
app = FastAPI(
    title="RAF ICD Extraction API",
    description="AI-powered ICD-10 code extraction from clinical documents",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://localhost:9090"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug middleware to log incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/extract-icd":
        print(f"\n🔍 DEBUG: Incoming request to {request.url.path}")
        print(f"   Method: {request.method}")
        print(f"   Headers: {dict(request.headers)}")
        content_type = request.headers.get("content-type", "")
        print(f"   Content-Type: {content_type}")
        if "boundary=" in content_type:
            boundary = content_type.split("boundary=")[1]
            print(f"   Boundary extracted: {boundary}")
    
    response = await call_next(request)
    return response

# Load master data and FAISS index at startup
print("Loading master data and FAISS index...")
try:
    icd10_master_df = pd.read_csv("data/icd10cm_2026.csv", dtype=str)
    icd9_master_df = pd.read_excel("data/valid_icd_9_codes.xlsx", dtype=str)
    gem_df = pd.read_csv("data/2015_I9gem.csv", dtype=str)
    
    # Normalize codes
    icd10_master_df = icd10_master_df.rename(columns={"code": "icd_code"})
    icd10_master_df["icd_code"] = icd10_master_df["icd_code"].str.replace(".", "", regex=False)
    icd9_master_df["icd_code"] = icd9_master_df["icd_code"].str.replace(".", "", regex=False)
    gem_df["icd9_code"] = gem_df["icd9_code"].astype(str)
    gem_df["icd10_code"] = gem_df["icd10_code"].astype(str)
    
    # Build lookup dictionaries
    icd10_code_to_desc = dict(zip(icd10_master_df['icd_code'], icd10_master_df['long_title']))
    icd10_code_to_billable = dict(zip(icd10_master_df['icd_code'], icd10_master_df['is_billable']))
    
    # ICD-9 uses 'icd_description' instead of 'long_title'
    icd9_code_to_desc = dict(zip(icd9_master_df['icd_code'], icd9_master_df['icd_description']))
    
    # Calculate billable ratio
    billable_count = (icd10_master_df['is_billable'] == '1').sum()
    total_count = len(icd10_master_df)
    billable_ratio = billable_count / total_count if total_count > 0 else 0.85
    
    # Pre-load FAISS index
    faiss_index = load_faiss_index()
    
    print("✅ Master data and FAISS index loaded successfully!")
except Exception as e:
    print(f"❌ Error loading master data: {e}")
    raise


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded file based on file type"""
    extension = filename.lower().split('.')[-1]
    
    if extension == 'pdf':
        text, used_ocr = extract_text_from_pdf(file_path)
        return text
    elif extension == 'txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif extension in ['doc', 'docx']:
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading Word document: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")


def process_single_document(file_path: str, filename: str) -> List[Dict]:
    """Process a single document and extract ICD codes"""
    
    print(f"\n🔍 Processing: {filename}")
    
    # Step 1: Extract and clean text
    raw_text = extract_text_from_file(file_path, filename)
    
    if not raw_text.strip():
        print(f"⚠️ No text extracted from {filename}")
        return []
    
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text_by_tokens(cleaned_text, max_tokens=200)
    print(f"   Extracted {len(chunks)} chunks from {filename}")
    
    # Step 2: LLM Semantic Extraction (Batch Processing)
    print(f"   Running LLM extraction...")
    batch_results = extract_icd_from_chunks_batch(chunks, batch_size=5)
    
    semantic_icd_list = []
    diagnosis_objects_list = []
    
    for semantic_codes, diagnoses in batch_results:
        semantic_icd_list.append(semantic_codes)
        diagnosis_objects_list.append(diagnoses)
    
    print(f"   Found {sum(len(codes) for codes in semantic_icd_list)} initial diagnoses")
    
    # Step 3: Validate ICD-10 codes
    validated_icd10_list = []
    mismatched_codes_list = []
    
    for semantic_codes in semantic_icd_list:
        matched_icd10, mismatched = validate_icd_codes(semantic_codes, icd10_master_df)
        validated_icd10_list.append(matched_icd10)
        mismatched_codes_list.append(mismatched)
    
    print(f"   Validated {sum(len(x) for x in validated_icd10_list)} ICD-10 codes")
    
    # Step 4: ICD-9 Fallback
    validated_icd9_list = []
    truly_invalid_codes_list = []
    
    for mismatched in mismatched_codes_list:
        matched_icd9, truly_invalid = validate_icd_codes(mismatched, icd9_master_df)
        validated_icd9_list.append(matched_icd9)
        truly_invalid_codes_list.append(truly_invalid)
    
    # Step 5: GEM Mapping (ICD-9 to ICD-10) with detailed tracking
    mapped_icd10_list = []
    gem_provenance = {}  # Track detailed GEM mapping info for each code
    
    for i, matched_icd9 in enumerate(validated_icd9_list):
        mapped_icd10 = []
        
        for icd9_code in matched_icd9:
            normalized = icd9_code.replace(".", "")
            
            # Get evidence for this ICD-9 code
            evidence_text = ""
            if i < len(diagnosis_objects_list):
                for diag in diagnosis_objects_list[i]:
                    if hasattr(diag, 'evidence_snippet'):
                        evidence_text = diag.evidence_snippet
                        break
            
            # Priority: approximate = 1
            approx_matches = gem_df[
                (gem_df["icd9_code"] == normalized) &
                (gem_df["approximate"] == "1")
            ]["icd10_code"].tolist()
            
            if not approx_matches:
                approx_matches = gem_df[
                    (gem_df["icd9_code"] == normalized) &
                    (gem_df["approximate"] == "0")
                ]["icd10_code"].tolist()
            
            if approx_matches:
                if len(approx_matches) > 1:
                    try:
                        icd9_desc = icd9_code_to_desc.get(normalized, "Unknown condition")
                        gem_result = select_best_icd10_from_gem_detailed(
                            icd9_code=icd9_code,
                            icd9_description=icd9_desc,
                            icd10_candidates=approx_matches,
                            icd10_descriptions=icd10_code_to_desc,
                            clinical_context=chunks[i] if i < len(chunks) else "",
                            clinical_evidence=evidence_text
                        )
                        if gem_result:
                            best_code = gem_result["selected_code"]
                            if best_code and best_code not in mapped_icd10:
                                mapped_icd10.append(best_code)
                                gem_provenance[best_code.replace(".", "")] = gem_result
                    except Exception as e:
                        print(f"⚠️  GEM selection error: {e}")
                        if approx_matches[0] not in mapped_icd10:
                            mapped_icd10.append(approx_matches[0])
                else:
                    if approx_matches[0] not in mapped_icd10:
                        mapped_icd10.append(approx_matches[0])
                        # Track single mapping provenance
                        gem_provenance[approx_matches[0].replace(".", "")] = {
                            "original_icd9_code": icd9_code,
                            "original_icd9_description": icd9_code_to_desc.get(normalized, ""),
                            "icd10_candidates": [approx_matches[0]],
                            "selected_code": approx_matches[0],
                            "selected_description": icd10_code_to_desc.get(approx_matches[0].replace(".", ""), ""),
                            "reasoning": f"Direct 1:1 GEM mapping from ICD-9 {icd9_code} to ICD-10 {approx_matches[0]}",
                            "evidence_snippet": evidence_text
                        }
        
        mapped_icd10_list.append(mapped_icd10)
    
    print(f"   Mapped {sum(len(x) for x in mapped_icd10_list)} ICD-9 → ICD-10 codes")
    
    # Step 6: FAISS + LLM Correction for invalid codes with detailed tracking
    corrected_icd10_list = []
    faiss_provenance = {}  # Track detailed FAISS correction info for each code
    total_truly_invalid = sum(len(x) for x in truly_invalid_codes_list)
    
    if total_truly_invalid > 0:
        print(f"   Correcting {total_truly_invalid} invalid codes using FAISS + LLM...")
        
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
                detailed_results = smart_result.get("detailed_results", [])
                
                # Store detailed provenance for each corrected code
                for detail in detailed_results:
                    if detail:
                        corrected_code = detail.get("llm2_valid_icd_code", "").replace(".", "")
                        faiss_provenance[corrected_code] = detail
                
                corrected_codes = list(corrected_codes_dict.values())
            
            corrected_icd10_list.append(corrected_codes)
    else:
        corrected_icd10_list = [[] for _ in semantic_icd_list]
    
    # Step 7: Global Reconciliation
    print(f"   Running global reconciliation...")
    
    # Build provenance dict for validated ICD-10 codes (directly extracted and valid)
    validated_provenance = {}
    for i, validated_codes in enumerate(validated_icd10_list):
        if i < len(diagnosis_objects_list):
            for code in validated_codes:
                normalized = code.replace(".", "")
                # Find matching diagnosis object
                for diag in diagnosis_objects_list[i]:
                    if diag.icd10.replace(".", "") == normalized:
                        validated_provenance[normalized] = {
                            "condition": diag.condition,
                            "evidence": diag.evidence_snippet,
                            "source": "direct_extraction"
                        }
                        break
    
    try:
        reconciled_icd_codes, reconciled_diagnoses = reconcile_diagnoses_globally(
            all_chunk_results=batch_results,
            chunks=chunks,
            max_retries=2
        )
        
        if reconciled_diagnoses:
            print(f"   ✅ Reconciled to {len(reconciled_icd_codes)} final codes")
            final_icd10_codes = reconciled_icd_codes
        else:
            # Fallback: combine all codes
            all_codes = []
            for validated, mapped, corrected in zip(validated_icd10_list, mapped_icd10_list, corrected_icd10_list):
                all_codes.extend(validated + mapped + corrected)
            final_icd10_codes = list(dict.fromkeys(all_codes))
            print(f"   Using fallback: {len(final_icd10_codes)} unique codes")
    except Exception as e:
        print(f"   Reconciliation failed: {e}")
        all_codes = []
        for validated, mapped, corrected in zip(validated_icd10_list, mapped_icd10_list, corrected_icd10_list):
            all_codes.extend(validated + mapped + corrected)
        final_icd10_codes = list(dict.fromkeys(all_codes))
    
    # Build final response with complete provenance
    icd_codes_response = build_icd_code_response_with_provenance(
        final_codes=final_icd10_codes,
        validated_icd10_codes=validated_provenance,
        gem_mapped_codes=gem_provenance,
        faiss_corrected_codes=faiss_provenance,
        icd10_code_to_desc=icd10_code_to_desc,
        icd10_code_to_billable=icd10_code_to_billable,
        chart_date=None  # TODO: Extract chart date from document if available
    )
    
    print(f"   ✅ Completed: {len(icd_codes_response)} final ICD codes")
    return icd_codes_response


@app.post("/extract-icd")
async def extract_icd(file: UploadFile = File(...)):
    """
    Extract ICD-10 codes from a single uploaded document
    """
    print(f"\n{'='*60}")
    print(f"📥 Received request to /extract-icd")
    print(f"   File parameter: {file}")
    if file:
        print(f"   Filename: {file.filename}")
        print(f"   Content-Type: {file.content_type}")
        print(f"   Size: {file.size if hasattr(file, 'size') else 'unknown'}")
    else:
        print(f"   ⚠️ File is None!")
    print(f"{'='*60}")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process document
        icd_codes = process_single_document(tmp_path, file.filename)
        
        # Clean up temp file
        os.remove(tmp_path)
        
        return JSONResponse(content={
            "icd_codes": icd_codes,
            "message": "Extraction successful",
            "file_name": file.filename
        })
        
    except Exception as e:
        print(f"❌ Error processing {file.filename}: {str(e)}")
        if 'tmp_path' in locals():
            try:
                os.remove(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAF ICD Extraction API",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAF ICD Extraction API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting RAF ICD Extraction FastAPI Service")
    print("="*60)
    print("📍 API URL: http://localhost:8500")
    print("📖 API Docs: http://localhost:8500/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "fastapi_service:app",
        host="0.0.0.0",
        port=8500,
        reload=True,
        log_level="info"
    )
