"""
LLM-based selection of best ICD-10 code when multiple mappings exist from ICD-9.
Returns detailed reasoning for transparency.
"""

from typing import Optional, List, Dict
import time
import json
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
    Returns only the selected code (backward compatibility).
    
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
    
    result = select_best_icd10_from_gem_detailed(
        icd9_code=icd9_code,
        icd9_description=icd9_description,
        icd10_candidates=icd10_candidates,
        icd10_descriptions=icd10_descriptions,
        clinical_context=clinical_context,
        clinical_evidence=clinical_evidence,
        max_retries=max_retries
    )
    
    return result["selected_code"] if result else None


def select_best_icd10_from_gem_detailed(
    icd9_code: str,
    icd9_description: str,
    icd10_candidates: List[str],
    icd10_descriptions: dict,
    clinical_context: str,
    clinical_evidence: str = "",
    max_retries: int = 2
) -> Optional[Dict]:
    """
    Use LLM to select the best ICD-10 code when multiple GEM mappings exist.
    Returns detailed information including reasoning.
    
    Args:
        icd9_code: The ICD-9 code (e.g., "1269")
        icd9_description: Description of the ICD-9 code
        icd10_candidates: List of possible ICD-10 codes from GEM mapping
        icd10_descriptions: Dictionary mapping ICD-10 codes to their descriptions
        clinical_context: The clinical note/chunk text
        clinical_evidence: Specific evidence snippet supporting this diagnosis
        max_retries: Number of retry attempts
    
    Returns:
        Dictionary with:
        - original_icd9_code: The original ICD-9 code
        - original_icd9_description: Description of ICD-9 code
        - icd10_candidates: List of candidate ICD-10 codes
        - selected_code: The selected ICD-10 code
        - selected_description: Description of selected code
        - reasoning: LLM's detailed reasoning
        - evidence_snippet: Clinical evidence used
        Or None if selection fails
    """
    
    # If only one candidate, return it immediately with simple reasoning
    if len(icd10_candidates) == 1:
        return {
            "original_icd9_code": icd9_code,
            "original_icd9_description": icd9_description,
            "icd10_candidates": icd10_candidates,
            "selected_code": icd10_candidates[0],
            "selected_description": icd10_descriptions.get(icd10_candidates[0], "Description not available"),
            "reasoning": f"ICD-9 code {icd9_code} ({icd9_description}) has only one ICD-10 mapping via GEM: {icd10_candidates[0]}. This is a direct 1:1 mapping with no ambiguity.",
            "evidence_snippet": clinical_evidence if clinical_evidence else "No specific evidence available"
        }
    
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
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                parsed = json.loads(response_text)
                selected_code = parsed.get("selected_code", "").strip()
                reasoning = parsed.get("reasoning", "")
            except json.JSONDecodeError:
                # Fallback: try to extract code from text
                selected_code = response_text
                reasoning = "LLM response was not in JSON format"
            
            # Validate that the selected code is in the candidates
            if selected_code in icd10_candidates:
                return {
                    "original_icd9_code": icd9_code,
                    "original_icd9_description": icd9_description,
                    "icd10_candidates": icd10_candidates,
                    "selected_code": selected_code,
                    "selected_description": icd10_descriptions.get(selected_code, "Description not available"),
                    "reasoning": reasoning,
                    "evidence_snippet": clinical_evidence if clinical_evidence else "No specific evidence available"
                }
            
            # Handle case where LLM returns code with decimal
            selected_normalized = selected_code.replace(".", "")
            if selected_normalized in icd10_candidates:
                return {
                    "original_icd9_code": icd9_code,
                    "original_icd9_description": icd9_description,
                    "icd10_candidates": icd10_candidates,
                    "selected_code": selected_normalized,
                    "selected_description": icd10_descriptions.get(selected_normalized, "Description not available"),
                    "reasoning": reasoning,
                    "evidence_snippet": clinical_evidence if clinical_evidence else "No specific evidence available"
                }
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Fallback: return first candidate with fallback reasoning
                return {
                    "original_icd9_code": icd9_code,
                    "original_icd9_description": icd9_description,
                    "icd10_candidates": icd10_candidates,
                    "selected_code": icd10_candidates[0],
                    "selected_description": icd10_descriptions.get(icd10_candidates[0], "Description not available"),
                    "reasoning": f"LLM selection failed after {max_retries} attempts. Defaulted to first candidate from GEM mapping. Error: {str(e)[:200]}",
                    "evidence_snippet": clinical_evidence if clinical_evidence else "No specific evidence available"
                }
            time.sleep(2)  # Backoff before retry
            continue
    
    # Final fallback
    return {
        "original_icd9_code": icd9_code,
        "original_icd9_description": icd9_description,
        "icd10_candidates": icd10_candidates,
        "selected_code": icd10_candidates[0],
        "selected_description": icd10_descriptions.get(icd10_candidates[0], "Description not available"),
        "reasoning": "LLM selection failed. Defaulted to first candidate from GEM mapping.",
        "evidence_snippet": clinical_evidence if clinical_evidence else "No specific evidence available"
    }
