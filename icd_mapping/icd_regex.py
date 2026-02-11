"""
Extract directly mentioned ICD-like codes using regex.
Returns a flat unique list without assuming ICD version.
"""

import re
from typing import List


# ICD-10-CM Pattern (more reliable)
# Letter (A-Z except U) + 2 digits + optional decimal + 1-4 alphanumeric
ICD10_PATTERN = r'\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?\b'

# ICD-9 Pattern (more restrictive to avoid page numbers)
# Must have decimal OR be 4-5 digits (not just 3 digits)
ICD9_PATTERN = r'\b(?:\d{3}\.\d{1,2}|\d{4,5})\b'


def extract_icd_codes(text: str) -> List[str]:
    """
    Extract unique ICD-like codes from text.
    Returns a flat list without version classification.
    
    ICD-10: Letter + 2 digits (e.g., E11.9, Z79.4)
    ICD-9: 3 digits with decimal OR 4-5 digits (e.g., 250.00, 4019)
    
    Note: Plain 3-digit numbers (105, 127) are excluded to avoid 
    false positives like page numbers.
    """

    # Extract ICD-10 codes
    icd10_matches = re.findall(ICD10_PATTERN, text)
    
    # Extract ICD-9 codes (more restrictive)
    icd9_matches = re.findall(ICD9_PATTERN, text)

    all_matches = icd10_matches + icd9_matches

    # Remove duplicates while preserving order
    seen = set()
    unique_codes = []

    for code in all_matches:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    return unique_codes
