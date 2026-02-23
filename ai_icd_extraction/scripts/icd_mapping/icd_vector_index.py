"""
FAISS-based semantic search for ICD codes.
Loads pre-built index and provides similarity search.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Tuple
import os

# Global variables for lazy loading
_faiss_index = None
_embeddings = None

def load_faiss_index():
    """
    Load FAISS index from disk (lazy loading).
    Only loads once, then reuses the loaded index.
    """
    global _faiss_index, _embeddings
    
    if _faiss_index is None:
        # Try multiple paths depending on where script is run from
        possible_paths = [
            "data/faiss_icd_index",  # When run from ai_icd_extraction/
            "ai_icd_extraction/data/faiss_icd_index",  # When run from project root
        ]
        
        index_path = None
        for path in possible_paths:
            if os.path.exists(path):
                index_path = path
                break
        
        if index_path is None:
            raise FileNotFoundError(
                f"FAISS index not found. Tried paths: {possible_paths}. "
                "Run 'python ai_icd_extraction/base_script/build_faiss_index.py' first "
                "or 'cd ai_icd_extraction && python base_script/build_faiss_index.py'"
            )
        
        # Initialize embeddings (must match the model used during index creation)
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load index
        _faiss_index = FAISS.load_local(
            index_path, 
            _embeddings,
            allow_dangerous_deserialization=True
        )
    
    return _faiss_index


def find_similar_icd_codes(
    condition_text: str, 
    top_k: int = 5,
    score_threshold: float = 0.5,
    billable_only: bool = True,
    billable_ratio: float = 0.85
) -> List[Tuple[str, str, float]]:
    """
    Find similar ICD codes using semantic search.
    
    Performance Optimization (Step 7):
    - Uses optimal search multiplier based on billable ratio
    - Reduces search space from 3x to ~1.2x (60% reduction)
    - Maintains same accuracy (still gets required billable codes)
    
    Args:
        condition_text: Medical condition description
        top_k: Number of similar codes to return
        score_threshold: Minimum similarity score (0-1)
        billable_only: If True, only return billable codes (is_billable='1')
        billable_ratio: Ratio of billable codes in dataset (default: 0.85)
    
    Returns:
        List of (code, description, score) tuples
        Example: [("E11.9", "Type 2 diabetes without complications", 0.92), ...]
    """
    
    faiss_index = load_faiss_index()
    
    # Step 7: Optimize search multiplier based on actual billable ratio
    # Old: search_k = top_k * 3 (always search 3x more)
    # New: search_k = top_k * (1 / billable_ratio)
    # Example: If 85% are billable, multiplier = 1.18 instead of 3
    if billable_only:
        search_multiplier = 1.0 / billable_ratio if billable_ratio > 0 else 3.0
        # Add small buffer for safety (10%)
        search_multiplier = search_multiplier * 1.1
        search_k = int(top_k * search_multiplier)
    else:
        search_k = top_k
    
    # Search for similar codes
    results = faiss_index.similarity_search_with_score(
        condition_text, 
        k=search_k
    )
    
    # Format results
    similar_codes = []
    for doc, score in results:
        # FAISS returns distance, convert to similarity (lower distance = higher similarity)
        # Normalize score to 0-1 range (approximate)
        similarity_score = 1.0 / (1.0 + score)
        
        if similarity_score >= score_threshold:
            code = doc.metadata["code"]
            description = doc.metadata["long_title"]
            is_billable = doc.metadata.get("is_billable", "1")  # Default to billable if not set
            
            # Filter out non-billable codes if requested
            if billable_only and is_billable != "1":
                continue
            
            similar_codes.append((code, description, similarity_score))
            
            # Stop if we have enough billable codes
            if len(similar_codes) >= top_k:
                break
    
    return similar_codes


def find_similar_by_invalid_code(
    invalid_code: str,
    condition_text: str,
    top_k: int = 5,
    billable_only: bool = True,
    billable_ratio: float = 0.85
) -> List[Tuple[str, str, float]]:
    """
    Find similar codes for an invalid/mismatched code.
    Only returns BILLABLE codes by default to prevent non-billable parent code mappings.
    
    Args:
        invalid_code: The invalid ICD code (e.g., "E11.999")
        condition_text: Medical condition description
        top_k: Number of candidates to return
        billable_only: If True, only return billable codes (prevents G20, etc.)
        billable_ratio: Ratio of billable codes (for search optimization)
    
    Returns:
        List of (code, description, score) tuples - BILLABLE CODES ONLY
    """
    
    # Use condition text for semantic search
    query = f"{invalid_code}: {condition_text}"
    
    return find_similar_icd_codes(
        query, 
        top_k=top_k, 
        billable_only=billable_only,
        billable_ratio=billable_ratio
    )
