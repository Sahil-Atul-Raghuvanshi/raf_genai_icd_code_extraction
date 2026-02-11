"""
Sentence-aware chunker that:
- Keeps sentences complete
- Handles ICD codes with decimal points
- Splits into ~200 token chunks
"""

import re
from typing import List


ICD_PATTERN = r'\b(?:[A-TV-Z][0-9][0-9](?:\.[0-9A-TV-Z]{1,4})?|\d{3}(?:\.\d{1,2})?)\b'

def protect_icd_codes(text: str) -> str:
    """
    Replace '.' inside ICD codes temporarily
    so sentence splitting does not break them.
    """
    return re.sub(ICD_PATTERN, lambda m: m.group(0).replace(".", "<DOT>"), text)


def restore_icd_codes(text: str) -> str:
    """
    Restore original ICD codes.
    """
    return text.replace("<DOT>", ".")


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences safely.
    """
    # Split only on period followed by space + capital letter
    sentences = re.split(r'(?<=[a-zA-Z0-9])\.\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_by_tokens(text: str, max_tokens: int = 200) -> List[str]:
    """
    Chunk text into ~max_tokens sized chunks.
    Keeps only full sentences.
    """

    protected_text = protect_icd_codes(text)
    sentences = split_into_sentences(protected_text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(sentence.split())

        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(restore_icd_codes(chunk_text))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(restore_icd_codes(chunk_text))

    return chunks
