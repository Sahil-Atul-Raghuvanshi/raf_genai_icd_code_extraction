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

    # Step 1: Protect ICD codes by replacing dots with placeholders
    # This prevents sentence splitting from breaking codes like "E11.9" into "E11" and "9"
    protected_text = protect_icd_codes(text)
    
    # Step 2: Split text into individual sentences
    sentences = split_into_sentences(protected_text)

    # Initialize chunking variables
    chunks = []           # Final list of text chunks to return
    current_chunk = []    # Sentences being accumulated for current chunk
    current_tokens = 0    # Token count for current chunk

    # Step 3: Group sentences into chunks while respecting token limit
    for sentence in sentences:
        # Count tokens in this sentence (simple word count)
        sentence_tokens = len(sentence.split())

        # Check if adding this sentence would exceed the max_tokens limit
        if current_tokens + sentence_tokens > max_tokens:
            # Current chunk is full, finalize it
            if current_chunk:
                # Join sentences with spaces and restore ICD code dots
                chunk_text = " ".join(current_chunk)
                chunks.append(restore_icd_codes(chunk_text))
            
            # Reset for next chunk
            current_chunk = []
            current_tokens = 0

        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Step 4: Add the last chunk if it contains any sentences
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(restore_icd_codes(chunk_text))

    return chunks
