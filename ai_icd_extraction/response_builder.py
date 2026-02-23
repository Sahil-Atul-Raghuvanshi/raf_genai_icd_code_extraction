"""
Enhanced process_single_document with detailed provenance tracking.
This module tracks the complete journey of each ICD code through the pipeline.
"""

from typing import List, Dict, Optional
from datetime import datetime


def build_icd_code_response_with_provenance(
    final_codes: List[str],
    validated_icd10_codes: Dict[str, Dict],  # {code: {condition, evidence, source}}
    gem_mapped_codes: Dict[str, Dict],  # {code: {original_icd9, reasoning, evidence}}
    faiss_corrected_codes: Dict[str, Dict],  # {code: {original_invalid, reasoning, evidence}}
    icd10_code_to_desc: dict,
    icd10_code_to_billable: dict,
    chart_date: str = None
) -> List[Dict]:
    """
    Build final response with complete provenance and reasoning for each code.
    
    Args:
        final_codes: List of final ICD-10 codes
        validated_icd10_codes: Dict of codes that were directly valid
        gem_mapped_codes: Dict of codes that came from ICD-9 → ICD-10 mapping
        faiss_corrected_codes: Dict of codes that were corrected via FAISS
        icd10_code_to_desc: Lookup dict for descriptions
        icd10_code_to_billable: Lookup dict for billable status
        chart_date: Optional chart date from document
    
    Returns:
        List of dictionaries with complete code information
    """
    
    icd_codes_response = []
    
    for code in final_codes:
        normalized = code.replace(".", "")
        description = icd10_code_to_desc.get(normalized, "Description not found")
        is_billable = icd10_code_to_billable.get(normalized, "Unknown")
        billable_status = "Yes" if is_billable == "1" else "No" if is_billable == "0" else "Unknown"
        
        # Determine provenance and reasoning
        reasoning = ""
        evidence = ""
        
        if normalized in validated_icd10_codes:
            # Code was directly extracted and validated
            info = validated_icd10_codes[normalized]
            evidence = info.get("evidence", "")
            reasoning = f"This ICD-10 code was directly extracted from the clinical documentation and validated against the ICD-10-CM 2026 codebook. Clinical condition: {info.get('condition', 'N/A')}. The code is valid and billable."
        
        elif normalized in gem_mapped_codes:
            # Code came from ICD-9 → ICD-10 mapping
            info = gem_mapped_codes[normalized]
            original_icd9 = info.get("original_icd9_code", "Unknown")
            icd9_desc = info.get("original_icd9_description", "")
            llm_reasoning = info.get("reasoning", "")
            candidates = info.get("icd10_candidates", [])
            evidence = info.get("evidence_snippet", "")
            
            reasoning = f"ORIGIN: Initially extracted as ICD-9 code {original_icd9} ({icd9_desc}). "
            
            if len(candidates) > 1:
                reasoning += f"MAPPING: GEM (General Equivalence Mappings) returned {len(candidates)} possible ICD-10 codes: {', '.join(candidates)}. "
                reasoning += f"LLM SELECTION: {llm_reasoning}"
            else:
                reasoning += f"MAPPING: Direct 1:1 GEM mapping to ICD-10 code {normalized}. {llm_reasoning}"
        
        elif normalized in faiss_corrected_codes:
            # Code was corrected via FAISS semantic search
            info = faiss_corrected_codes[normalized]
            original_invalid = info.get("llm1_icd_code", "Unknown")
            condition = info.get("llm1_description", "")
            llm_reasoning = info.get("reasoning", "")
            top_5 = info.get("top_5_similar_codes", {})
            evidence = info.get("evidence_snippet", "")
            
            reasoning = f"ORIGIN: Initially extracted as invalid code '{original_invalid}' for condition: {condition}. "
            reasoning += f"CORRECTION: Code was invalid (not found in ICD-10-CM codebook). Used FAISS semantic search to find top 5 similar valid codes: {', '.join(list(top_5.keys())[:5])}. "
            reasoning += f"LLM SELECTION: {llm_reasoning}"
        
        else:
            # Fallback - code source unknown
            reasoning = "Code was extracted and validated through the standard pipeline."
            evidence = "Evidence not available"
        
        icd_codes_response.append({
            "icd_code": code,
            "icd_description": description,
            "is_billable": billable_status,
            "evidence_snippet": evidence,
            "llm_reasoning": reasoning,
            "chart_date": chart_date if chart_date else datetime.now().strftime("%Y-%m-%d")
        })
    
    return icd_codes_response
