"""
Intelligent filtering for ICD code corrections.
Determines which invalid codes need expensive LLM correction vs simple fixes.
"""

import re
from typing import Optional, Tuple


def is_simple_format_error(invalid_code: str) -> bool:
    """
    Check if the invalid code is just a simple formatting error.
    
    Common format errors:
    - Missing decimal: E119 → E11.9
    - Extra spaces: E11 .9 → E11.9
    - Wrong case: e11.9 → E11.9
    - Extra characters: E11.9x → E11.9
    
    Args:
        invalid_code: The invalid ICD code
    
    Returns:
        True if it's a simple format error that can be fixed instantly
    """
    if not invalid_code or len(invalid_code) < 3:
        return False
    
    # Check for missing decimal (e.g., E119, I10)
    # ICD-10 format: Letter + 2 digits [+ decimal + 1-2 more digits]
    pattern = r'^[A-Z]\d{3,5}$'
    if re.match(pattern, invalid_code, re.IGNORECASE):
        return True  # Missing decimal
    
    # Check for extra spaces
    if ' ' in invalid_code:
        return True
    
    # Check for wrong case (lowercase letter)
    if invalid_code[0].islower():
        return True
    
    # Check for trailing non-alphanumeric characters
    if re.search(r'[^A-Z0-9.]$', invalid_code, re.IGNORECASE):
        return True
    
    return False


def fix_format(invalid_code: str, icd10_master_df=None) -> Optional[str]:
    """
    Apply instant format fixes to invalid codes.
    
    Args:
        invalid_code: The invalid ICD code
        icd10_master_df: Optional ICD-10 master dataframe for validation
    
    Returns:
        Corrected code or None if can't be fixed
    """
    if not invalid_code:
        return None
    
    # Clean up
    code = invalid_code.strip().upper()
    
    # Remove extra spaces
    code = code.replace(' ', '')
    
    # Remove trailing non-alphanumeric characters
    code = re.sub(r'[^A-Z0-9.]+$', '', code)
    
    # Add missing decimal for 4+ digit codes
    # E119 → E11.9, I10 stays I10
    if re.match(r'^[A-Z]\d{4,5}$', code):
        # Insert decimal after 3rd character
        code = code[:3] + '.' + code[3:]
    
    # Validate format
    if not re.match(r'^[A-Z]\d{2,3}(\.\d{1,4})?$', code):
        return None
    
    # If master dataframe provided, validate against it
    if icd10_master_df is not None:
        if code in icd10_master_df['icd_code'].values:
            return code
        else:
            return None  # Format fixed but code doesn't exist
    
    return code


def calculate_condition_confidence(condition_text: str, evidence_snippet: str = "") -> float:
    """
    Calculate confidence score for extracted condition.
    
    Higher confidence means:
    - Specific condition name (not vague)
    - Strong evidence from clinical text
    - Standard medical terminology
    
    Args:
        condition_text: The extracted medical condition
        evidence_snippet: Supporting evidence from clinical text
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not condition_text or len(condition_text.strip()) < 5:
        return 0.0
    
    score = 0.5  # Base score
    
    # Bonus for specific terms (not vague)
    vague_terms = ['unspecified', 'unknown', 'uncertain', 'unclear', 'possible', 'suspected']
    if not any(term in condition_text.lower() for term in vague_terms):
        score += 0.2
    
    # Bonus for specific medical details
    specific_indicators = [
        'type 1', 'type 2', 'acute', 'chronic', 'severe', 'mild', 'moderate',
        'left', 'right', 'bilateral', 'stage', 'class', 'grade', 'with complications',
        'without complications'
    ]
    if any(indicator in condition_text.lower() for indicator in specific_indicators):
        score += 0.2
    
    # Bonus for evidence snippet
    if evidence_snippet and len(evidence_snippet) > 10:
        score += 0.1
    
    # Penalty for very short condition text
    if len(condition_text) < 15:
        score -= 0.2
    
    # Penalty for generic terms
    generic_terms = ['disease', 'disorder', 'condition', 'syndrome']
    generic_only = all(
        term in condition_text.lower() 
        for term in generic_terms 
        if term in condition_text.lower()
    )
    if generic_only and len(condition_text.split()) < 4:
        score -= 0.2
    
    # Clamp between 0.0 and 1.0
    return max(0.0, min(1.0, score))


def should_correct_code(
    invalid_code: str,
    condition_text: str,
    evidence_snippet: str = "",
    icd10_master_df=None,
    confidence_threshold: float = 0.4
) -> Tuple[bool, Optional[str], str]:
    """
    Determine if an invalid code needs expensive LLM correction.
    
    Decision logic:
    1. If simple format error → Apply instant fix
    2. If low confidence condition → Skip correction
    3. Otherwise → Needs LLM correction
    
    Args:
        invalid_code: The invalid ICD code
        condition_text: Medical condition description
        evidence_snippet: Supporting evidence from clinical text
        icd10_master_df: ICD-10 master dataframe for validation
        confidence_threshold: Minimum confidence to attempt correction
    
    Returns:
        Tuple of (needs_llm_correction, instant_fix_code, reason)
        - needs_llm_correction: True if LLM correction needed
        - instant_fix_code: Corrected code if instant fix worked, else None
        - reason: Human-readable reason for the decision
    """
    
    # Check 1: Simple format error?
    if is_simple_format_error(invalid_code):
        fixed_code = fix_format(invalid_code, icd10_master_df)
        if fixed_code:
            return False, fixed_code, "instant_fix"
        else:
            return False, None, "format_fix_failed"
    
    # Check 2: Low confidence condition?
    confidence = calculate_condition_confidence(condition_text, evidence_snippet)
    if confidence < confidence_threshold:
        return False, None, f"low_confidence_{confidence:.2f}"
    
    # Check 3: Too far from valid format?
    if not re.match(r'^[A-Za-z]\d{2,5}\.?\d{0,4}[A-Za-z]?$', invalid_code):
        return False, None, "invalid_format"
    
    # Default: Needs LLM correction
    return True, None, "llm_required"


def filter_codes_for_correction(
    invalid_codes: list,
    condition_texts: list,
    evidence_snippets: list = None,
    icd10_master_df=None,
    confidence_threshold: float = 0.4
) -> dict:
    """
    Filter invalid codes into categories: instant_fix, needs_llm, skip.
    
    Args:
        invalid_codes: List of invalid ICD codes
        condition_texts: List of condition descriptions
        evidence_snippets: Optional list of evidence snippets
        icd10_master_df: ICD-10 master dataframe for validation
        confidence_threshold: Minimum confidence to attempt correction
    
    Returns:
        Dictionary with:
        - instant_fixes: {invalid_code: fixed_code}
        - needs_llm: [(invalid_code, condition, evidence)]
        - skipped: {invalid_code: reason}
        - stats: correction statistics
    """
    
    if evidence_snippets is None:
        evidence_snippets = [""] * len(invalid_codes)
    
    instant_fixes = {}
    needs_llm = []
    skipped = {}
    
    for code, condition, evidence in zip(invalid_codes, condition_texts, evidence_snippets):
        needs_correction, fixed_code, reason = should_correct_code(
            invalid_code=code,
            condition_text=condition,
            evidence_snippet=evidence,
            icd10_master_df=icd10_master_df,
            confidence_threshold=confidence_threshold
        )
        
        if fixed_code:
            # Instant fix applied
            instant_fixes[code] = fixed_code
        elif needs_correction:
            # Needs LLM correction
            needs_llm.append((code, condition, evidence))
        else:
            # Skip correction
            skipped[code] = reason
    
    # Calculate statistics
    total = len(invalid_codes)
    stats = {
        "total_invalid": total,
        "instant_fixes": len(instant_fixes),
        "needs_llm": len(needs_llm),
        "skipped": len(skipped),
        "llm_calls_saved": len(instant_fixes) + len(skipped),
        "time_saved_estimate": (len(instant_fixes) + len(skipped)) * 4,  # ~4s per LLM call
        "cost_reduction_pct": ((len(instant_fixes) + len(skipped)) / total * 100) if total > 0 else 0
    }
    
    return {
        "instant_fixes": instant_fixes,
        "needs_llm": needs_llm,
        "skipped": skipped,
        "stats": stats
    }
