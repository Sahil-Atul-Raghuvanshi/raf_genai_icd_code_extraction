from pydantic import BaseModel
from typing import List

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
