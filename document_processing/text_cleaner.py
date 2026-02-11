"""
Basic text cleaning and normalization.
"""

import re


def clean_text(text: str) -> str:
    """
    Cleans raw extracted text.
    """

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove repeated page numbers
    text = re.sub(r"Page \d+ of \d+", "", text)

    # Remove standalone page numbers
    text = re.sub(r"\n\d+\n", "\n", text)

    return text.strip()
