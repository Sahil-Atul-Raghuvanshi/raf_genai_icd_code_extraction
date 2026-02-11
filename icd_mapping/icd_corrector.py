"""
LLM-based ICD code correction using semantic search candidates.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import GOOGLE_API_KEY
from .icd_vector_index import find_similar_by_invalid_code
from typing import Optional, Dict, List, Tuple
import time

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    request_timeout=30
)

# Prompt template for code correction
CORRECTION_PROMPT = PromptTemplate(
    input_variables=["invalid_code", "condition", "candidates"],
    template="""You are a certified medical coder specializing in ICD-10-CM coding.

CRITICAL RULES:
1. You MUST select a BILLABLE ICD-10-CM code (ALL candidates provided are billable)
2. You MUST NOT change the clinical meaning or severity
3. You MUST NOT map acute codes to sequela codes or vice versa
4. You MUST NOT downgrade specificity level

The extracted ICD-10 code '{invalid_code}' is INVALID.
The clinical condition is: '{condition}'

Here are the top 5 most similar VALID and BILLABLE ICD-10 codes:
{candidates}

Task:
Select the SINGLE ICD-10 code that BEST matches the condition.

Selection Criteria (in priority order):
1. CLINICAL ACCURACY - Must match the exact clinical meaning
2. SPECIFICITY - Choose the most specific code that matches the documentation
3. SEVERITY - Must preserve documented severity level (mild, moderate, severe, class I/II/III)
4. LATERALITY - Must preserve side specification (left, right, bilateral, unilateral)
5. EPISODE - Must preserve temporal classification (acute, chronic, sequela, initial/subsequent encounter)
6. ANATOMICAL SITE - Must preserve specific body location if documented
7. TYPE/SUBTYPE - Must preserve disease type, complication type, or subclassification

FORBIDDEN Actions:
❌ DO NOT map to parent/non-specific codes
❌ DO NOT change temporal classification (acute ↔ chronic ↔ sequela)
❌ DO NOT change disease category or system
❌ DO NOT downgrade severity or specificity
❌ DO NOT change laterality or anatomical location
❌ DO NOT substitute "unspecified" codes when specific codes match

Return ONLY the ICD-10 code (e.g., E11.9), nothing else.
Do NOT include periods, explanations, or any other text.
"""
)


def correct_invalid_code_detailed(
    invalid_code: str,
    condition_text: str,
    max_retries: int = 2
) -> Optional[Dict]:
    """
    Correct an invalid ICD code using semantic search + LLM.
    Returns detailed information about the correction process.
    
    Args:
        invalid_code: The invalid ICD code (LLM1 output)
        condition_text: Medical condition description (LLM1 output)
        max_retries: Number of LLM retry attempts
    
    Returns:
        Dictionary with:
        - llm1_icd_code: Original invalid code
        - llm1_description: Original condition text
        - top_5_similar_codes: List of (code, description, score) tuples
        - llm2_valid_icd_code: Corrected code
        - llm2_valid_description: Description of corrected code
        Or None if correction fails
    """
    
    # Step 1: Get semantic candidates from FAISS
    try:
        candidates = find_similar_by_invalid_code(
            invalid_code=invalid_code,
            condition_text=condition_text,
            top_k=5
        )
    except Exception as e:
        print(f"Error finding similar codes: {e}")
        return None
    
    if not candidates:
        return None
    
    # Format candidates for prompt
    candidates_text = "\n".join([
        f"- {code}: {description} (similarity: {score:.2f})"
        for code, description, score in candidates
    ])
    
    # Step 2: Use LLM to select best match
    corrected_code = None
    for attempt in range(max_retries):
        try:
            prompt = CORRECTION_PROMPT.format(
                invalid_code=invalid_code,
                condition=condition_text,
                candidates=candidates_text
            )
            
            response = llm.invoke(prompt)
            corrected_code = response.content.strip()
            
            # Validate format (basic check)
            if len(corrected_code) >= 3 and corrected_code[0].isalpha():
                break
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Fallback: return highest scoring candidate
                corrected_code = candidates[0][0]
            time.sleep(2)  # Backoff before retry
            continue
    
    if not corrected_code:
        corrected_code = candidates[0][0]  # Final fallback
    
    # Get description for corrected code
    corrected_description = None
    for code, desc, score in candidates:
        if code == corrected_code:
            corrected_description = desc
            break
    
    if not corrected_description:
        corrected_description = candidates[0][1]
    
    # Build top 5 dictionary
    top_5_dict = {
        code: description 
        for code, description, score in candidates
    }
    
    return {
        "llm1_icd_code": invalid_code,
        "llm1_description": condition_text,
        "top_5_similar_codes": top_5_dict,
        "llm2_valid_icd_code": corrected_code,
        "llm2_valid_description": corrected_description
    }


def correct_invalid_code(
    invalid_code: str,
    condition_text: str,
    max_retries: int = 2
) -> Optional[str]:
    """
    Correct an invalid ICD code using semantic search + LLM.
    Returns only the corrected code (for backward compatibility).
    
    Args:
        invalid_code: The invalid ICD code
        condition_text: Medical condition description
        max_retries: Number of LLM retry attempts
    
    Returns:
        Corrected ICD code or None if correction fails
    """
    
    result = correct_invalid_code_detailed(invalid_code, condition_text, max_retries)
    return result["llm2_valid_icd_code"] if result else None


def correct_multiple_codes(
    invalid_codes: list,
    condition_texts: list
) -> dict:
    """
    Correct multiple invalid codes with their conditions.
    
    Args:
        invalid_codes: List of invalid ICD codes
        condition_texts: List of corresponding condition descriptions
    
    Returns:
        Dictionary mapping {invalid_code: corrected_code}
    """
    
    corrections = {}
    
    for i, (invalid_code, condition) in enumerate(zip(invalid_codes, condition_texts)):
        # Rate limiting
        if i > 0 and i % 5 == 0:
            time.sleep(1)
        
        corrected = correct_invalid_code(invalid_code, condition)
        if corrected:
            corrections[invalid_code] = corrected
    
    return corrections

