"""
Validate regex-extracted ICD codes against master ICD table.
Handles dot vs non-dot formatting differences.
"""

import pandas as pd
from typing import List, Tuple


def normalize_icd(code: str) -> str:
    """
    Remove period from ICD code for comparison.
    Example: E11.22 -> E1122
    """
    return code.replace(".", "").upper()


def validate_icd_codes(
    regex_codes: List[str],
    icd_master_df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Returns:
        matched_codes (with original format)
        mismatched_codes (with original format)
    """

    # Normalize master codes
    master_codes = set(icd_master_df["icd_code"].astype(str).str.upper())

    matched = []
    mismatched = []

    for code in regex_codes:
        normalized = normalize_icd(code)

        if normalized in master_codes:
            matched.append(code)
        else:
            mismatched.append(code)

    return matched, mismatched
