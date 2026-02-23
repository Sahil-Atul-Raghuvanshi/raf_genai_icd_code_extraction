from pydantic import BaseModel
from typing import List, Optional

class Diagnosis(BaseModel):
    condition: str
    icd10: str
    evidence_snippet: str

class ICDLLMResponse(BaseModel):
    diagnoses: List[Diagnosis]

class BatchChunkResult(BaseModel):
    chunk_number: int
    diagnoses: List[Diagnosis]

class BatchICDResponse(BaseModel):
    results: List[BatchChunkResult]

# Schema for global reconciliation (PASS 2)
class ReconciledDiagnosis(BaseModel):
    condition: str
    icd10: str
    evidence_snippet: str
    source_chunks: List[int]
    reasoning: str

class GlobalReconciliationResponse(BaseModel):
    reconciled_diagnoses: List[ReconciledDiagnosis]

# Schema for GEM selection reasoning
class GEMSelectionResult(BaseModel):
    original_icd9_code: str
    original_icd9_description: str
    selected_icd10_code: str
    selected_icd10_description: str
    icd10_candidates: List[str]
    reasoning: str
    evidence_snippet: str

# Schema for FAISS correction reasoning
class FAISSCorrectionResult(BaseModel):
    original_invalid_code: str
    condition: str
    top_5_candidates: dict  # {code: description}
    selected_code: str
    selected_description: str
    reasoning: str
    evidence_snippet: str
