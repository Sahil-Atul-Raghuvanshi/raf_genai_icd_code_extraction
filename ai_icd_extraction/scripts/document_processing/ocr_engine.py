"""
OCR engine for scanned PDFs using Tesseract.
"""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from typing import List


def extract_text_from_scanned_pdf(file_path: str) -> str:
    """
    Converts scanned PDF pages to images and runs OCR.
    """

    images: List[Image.Image] = convert_from_path(file_path)
    full_text = ""

    for img in images:
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"

    return full_text
