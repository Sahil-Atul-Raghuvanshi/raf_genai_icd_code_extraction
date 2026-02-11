"""
LLM-based selection of best ICD-10 code when multiple mappings exist from ICD-9.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import GOOGLE_API_KEY
from typing import Optional, List
import time
import pandas as pd

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    request_timeout=30
)

# Prompt template for selecting best ICD-10 code from multiple mappings
GEM_SELECTION_PROMPT = PromptTemplate(
    input_variables=["icd9_code", "icd9_description", "icd10_candidates", "clinical_evidence", "clinical_context"],
    template="""You are a certified medical coder specializing in ICD code mapping.

Task: Select the SINGLE MOST APPROPRIATE ICD-10 code for the given ICD-9 code based on clinical context.

**ICD-9 Code:** {icd9_code}
**ICD-9 Description:** {icd9_description}

**Available ICD-10 Code Options (from GEM mapping):**
{icd10_candidates}

**Clinical Evidence (Direct Quote from Note):**
{clinical_evidence}

**Full Clinical Context from Patient Note:**
{clinical_context}

Instructions:
1. Read the clinical evidence and context carefully
2. Consider the specific details mentioned (laterality, severity, complications, etc.)
3. Select the ICD-10 code that BEST matches the clinical documentation
4. Choose the most SPECIFIC code that is supported by the documentation
5. Prioritize evidence from the clinical evidence snippet, use full context for additional details

Return ONLY the ICD-10 code (e.g., B768), nothing else.
Do NOT include explanations, descriptions, or any other text.
"""
)


def select_best_icd10_from_gem(
    icd9_code: str,
    icd9_description: str,
    icd10_candidates: List[str],
    icd10_descriptions: dict,
    clinical_context: str,
    clinical_evidence: str = "",
    max_retries: int = 2
) -> Optional[str]:
    """
    Use LLM to select the best ICD-10 code when multiple GEM mappings exist.
    
    Args:
        icd9_code: The ICD-9 code (e.g., "1269")
        icd9_description: Description of the ICD-9 code
        icd10_candidates: List of possible ICD-10 codes from GEM mapping
        icd10_descriptions: Dictionary mapping ICD-10 codes to their descriptions
        clinical_context: The clinical note/chunk text
        clinical_evidence: Specific evidence snippet supporting this diagnosis
        max_retries: Number of retry attempts
    
    Returns:
        Selected ICD-10 code or None if selection fails
    """
    
    # If only one candidate, return it immediately
    if len(icd10_candidates) == 1:
        return icd10_candidates[0]
    
    # Format candidates for prompt
    candidates_text = "\n".join([
        f"- {code}: {icd10_descriptions.get(code, 'Description not available')}"
        for code in icd10_candidates
    ])
    
    # Try to get LLM selection
    for attempt in range(max_retries):
        try:
            prompt = GEM_SELECTION_PROMPT.format(
                icd9_code=icd9_code,
                icd9_description=icd9_description,
                icd10_candidates=candidates_text,
                clinical_evidence=clinical_evidence if clinical_evidence else "Not available",
                clinical_context=clinical_context[:1000]  # Limit context to 1000 chars
            )
            
            response = llm.invoke(prompt)
            selected_code = response.content.strip()
            
            # Validate that the selected code is in the candidates
            if selected_code in icd10_candidates:
                return selected_code
            
            # Handle case where LLM returns code with decimal
            selected_normalized = selected_code.replace(".", "")
            if selected_normalized in icd10_candidates:
                return selected_normalized
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Fallback: return first candidate
                return icd10_candidates[0]
            time.sleep(2)  # Backoff before retry
            continue
    
    # Final fallback
    return icd10_candidates[0]


def select_multiple_gem_mappings(
    icd9_codes: List[str],
    icd9_descriptions: dict,
    gem_mappings: dict,
    icd10_master_df: pd.DataFrame,
    clinical_context: str
) -> dict:
    """
    Process multiple ICD-9 codes and select best ICD-10 for each.
    
    Args:
        icd9_codes: List of ICD-9 codes to process
        icd9_descriptions: Dictionary of ICD-9 code descriptions
        gem_mappings: Dictionary {icd9_code: [list of icd10_codes]}
        icd10_master_df: DataFrame with ICD-10 codes and descriptions
        clinical_context: Clinical note text
    
    Returns:
        Dictionary {icd9_code: selected_icd10_code}
    """
    
    # Create ICD-10 description lookup
    icd10_desc_map = {}
    if 'icd_code' in icd10_master_df.columns and 'long_title' in icd10_master_df.columns:
        for _, row in icd10_master_df.iterrows():
            icd10_desc_map[row['icd_code']] = row['long_title']
    
    selections = {}
    
    for icd9_code in icd9_codes:
        candidates = gem_mappings.get(icd9_code, [])
        
        if not candidates:
            continue
        
        # Get ICD-9 description
        icd9_desc = icd9_descriptions.get(icd9_code, "Unknown condition")
        
        # Select best ICD-10 code
        selected = select_best_icd10_from_gem(
            icd9_code=icd9_code,
            icd9_description=icd9_desc,
            icd10_candidates=candidates,
            icd10_descriptions=icd10_desc_map,
            clinical_context=clinical_context
        )
        
        if selected:
            selections[icd9_code] = selected
    
    return selections
