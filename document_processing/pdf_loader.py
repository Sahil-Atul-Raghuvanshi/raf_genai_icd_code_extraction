"""
PDF loading and text extraction.
Automatically detects if OCR is required.
"""

import fitz  # PyMuPDF
from typing import Tuple
from .ocr_engine import extract_text_from_scanned_pdf


def is_text_valid(text: str) -> bool:
    if not text.strip():
        return False
    
    words = text.split()
    if len(words) < 50:
        return False
    
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.4


def extract_text_from_pdf(file_path: str) -> Tuple[str, bool]:

    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()

    if not is_text_valid(full_text):
        ocr_text = extract_text_from_scanned_pdf(file_path)
        return ocr_text, True

    return full_text, False
