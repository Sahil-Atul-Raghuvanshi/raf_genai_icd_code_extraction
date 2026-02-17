"""
LLM-based selection of best ICD-10 code when multiple mappings exist from ICD-9.
Uses shared LLM and prompts from clinical_extraction module for consistency.
"""

from typing import Optional, List
import time
import pandas as pd

# Import shared LLM and prompt from clinical_extraction module
from ai_icd_extraction.scripts.clinical_extraction.chain import llm_gem
from ai_icd_extraction.scripts.clinical_extraction.prompts import GEM_SELECTION_PROMPT


def select_best_icd10_from_gem(
    icd9_code: str,
    icd9_description: str,
    icd10_candidates: List[str],
    icd10_descriptions: dict,
    clinical_context: str,
    clinical_evidence: str = "",
    max_retries: int = 2
) -> Optional[str]:
    """
    Use LLM to select the best ICD-10 code when multiple GEM mappings exist.
    
    Args:
        icd9_code: The ICD-9 code (e.g., "1269")
        icd9_description: Description of the ICD-9 code
        icd10_candidates: List of possible ICD-10 codes from GEM mapping
        icd10_descriptions: Dictionary mapping ICD-10 codes to their descriptions
        clinical_context: The clinical note/chunk text
        clinical_evidence: Specific evidence snippet supporting this diagnosis
        max_retries: Number of retry attempts
    
    Returns:
        Selected ICD-10 code or None if selection fails
    """
    
    # If only one candidate, return it immediately
    if len(icd10_candidates) == 1:
        return icd10_candidates[0]
    
    # Format candidates for prompt
    candidates_text = "\n".join([
        f"- {code}: {icd10_descriptions.get(code, 'Description not available')}"
        for code in icd10_candidates
    ])
    
    # Try to get LLM selection using shared LLM from clinical_extraction.chain
    for attempt in range(max_retries):
        try:
            prompt = GEM_SELECTION_PROMPT.format(
                icd9_code=icd9_code,
                icd9_description=icd9_description,
                icd10_candidates=candidates_text,
                clinical_evidence=clinical_evidence if clinical_evidence else "Not available",
                clinical_context=clinical_context[:1000]  # Limit context to 1000 chars
            )
            
            # Use shared LLM instance from clinical_extraction module
            response = llm_gem.invoke(prompt)
            selected_code = response.content.strip()
            
            # Validate that the selected code is in the candidates
            if selected_code in icd10_candidates:
                return selected_code
            
            # Handle case where LLM returns code with decimal
            selected_normalized = selected_code.replace(".", "")
            if selected_normalized in icd10_candidates:
                return selected_normalized
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Fallback: return first candidate
                return icd10_candidates[0]
            time.sleep(2)  # Backoff before retry
            continue
    
    # Final fallback
    return icd10_candidates[0]
