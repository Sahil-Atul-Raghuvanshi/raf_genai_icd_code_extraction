from langchain_core.prompts import PromptTemplate

ICD_SEMANTIC_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template="""
You are a certified medical coder using the official 2026 ICD-10-CM codes from CMS.

Analyze the following clinical text and extract confirmed chronic medical conditions.

Instructions:
- Use ONLY valid ICD-10-CM codes from the 2026 CMS classification system (https://www.cms.gov/medicare/coding-billing/icd-10-codes)
- Extract diagnoses even if ICD code is not explicitly written.
- Assign the most specific billable ICD-10-CM code that accurately describes each condition.
- Write the CONDITION in detail matching the ICD-10-CM long description format (e.g., "Type 2 diabetes mellitus without complications" not just "diabetes").
- Include an EVIDENCE_SNIPPET: Extract the exact text from the clinical note that supports this diagnosis (verbatim quote, 5-20 words).
- Ignore negated conditions (e.g., "no history of MI").
- Ignore ruled-out, suspected, family history.
- Ignore labs, vitals, medications.
- Only return confirmed chronic diagnoses.
- Return ICD-10-CM codes only (format: letter + 2 digits + optional decimal, e.g., E11.9, B18.2).
- IMPORTANT: Return ONE ICD code per diagnosis entry. Do NOT combine multiple codes in one string.
- Each diagnosis should have exactly ONE condition, ONE icd10 code, and ONE evidence_snippet.
- If multiple conditions exist, create separate diagnosis entries for each.
- If no condition found, return empty list.
- Output STRICT JSON.

Clinical Text:
{chunk}

Output format:
{{
  "diagnoses": [
    {{
      "condition": "Chronic hepatitis C",
      "icd10": "B18.2",
      "evidence_snippet": "Patient has history of hepatitis C infection, chronic"
    }},
    {{
      "condition": "Alcoholic cirrhosis of liver",
      "icd10": "K70.30",
      "evidence_snippet": "Cirrhosis secondary to chronic alcohol use"
    }}
  ]
}}

Key Requirements:
1. CONDITION must be detailed and match ICD-10-CM long description style
2. EVIDENCE_SNIPPET must be exact text from the clinical note (verbatim quote)
3. ONE code per diagnosis (do NOT combine codes like "B18.2, K70.30")

WRONG format (do NOT do this):
{{
  "diagnoses": [
    {{
      "condition": "Hepatitis C and cirrhosis",
      "icd10": "B18.2, K70.30",
      "evidence_snippet": "Patient has liver disease"
    }}
  ]
}}
"""
)

ICD_SEMANTIC_BATCH_PROMPT = PromptTemplate(
    input_variables=["chunks_text", "num_chunks"],
    template="""
You are a certified medical coder using the official 2026 ICD-10-CM codes from CMS.

Analyze the following {num_chunks} clinical text chunks and extract confirmed chronic medical conditions from EACH chunk separately.

CRITICAL: Process each chunk independently and return results in the SAME ORDER.

{chunks_text}

For EACH chunk, return:
- condition: Detailed description matching ICD-10-CM long description
- icd10: Single ICD-10-CM code (billable, specific)
- evidence_snippet: Exact quote from that chunk (5-20 words)

Rules:
- Use ONLY valid 2026 ICD-10-CM codes
- Return ONE code per diagnosis (do NOT combine codes)
- Include evidence_snippet as verbatim quote from text
- If no conditions in a chunk, return empty diagnoses list for that chunk
- Ignore negated, ruled-out, suspected conditions

Output format:
{{
  "results": [
    {{
      "chunk_number": 1,
      "diagnoses": [
        {{
          "condition": "Type 2 diabetes mellitus without complications",
          "icd10": "E11.9",
          "evidence_snippet": "history of type 2 diabetes, well controlled"
        }}
      ]
    }},
    {{
      "chunk_number": 2,
      "diagnoses": [
        {{
          "condition": "Essential hypertension",
          "icd10": "I10",
          "evidence_snippet": "patient has longstanding hypertension"
        }}
      ]
    }}
  ]
}}

CRITICAL: Return exactly {num_chunks} results, one for each chunk, in the same order.
If a chunk has no conditions, include it with empty diagnoses list.
"""
)
