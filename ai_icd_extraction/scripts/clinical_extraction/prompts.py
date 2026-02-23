from langchain_core.prompts import PromptTemplate

ICD_SEMANTIC_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template="""
You are a medical coder using 2026 ICD-10-CM codes.

Extract CONFIRMED diagnoses ONLY. Apply these filters sequentially:

MANDATORY DECISION ALGORITHM:

1. CERTAINTY FILTER
EXCLUDE if statement contains:
Uncertainty: possible, may, suspected, likely, cannot exclude, uncertain
Differential: versus, vs, rule out, R/O, concerning for, question of
Negation: no, denies, absent, negative for

INCLUDE if: confirmed, history of, diagnosed with, known, established

2. ENTITY TYPE FILTER
INCLUDE:
Named diseases (diabetes, pneumonia, stroke, fracture, infarct)
Disease nomenclature (-itis, -osis, -oma, -pathy)
Sequela with disease term (old infarct, prior MI, history of stroke)

EXCLUDE:
Descriptors without disease name:
  Structural: changes, atrophy, calcification, prominence, volume loss
  Process: small vessel disease, microvascular changes, ischemic changes
  Imaging: signal abnormality, degenerative changes, chronic changes
Pattern: [adjective] + [structure] without disease entity

3. SYMPTOM VS DISEASE
If symptom + underlying disease present: Code disease ONLY
If symptom alone: Code symptom
Symptoms: pain, fever, cough, nausea, vertigo, dizziness, fatigue
Check for causal relationship (symptom "due to/from" disease)

4. TEMPORAL PRIORITY (Within Chunk)
If same disease mentioned multiple times in this chunk: Keep highest priority
Priority: Sequela > Chronic > Acute

Different episodes of same disease type: Keep ALL
Same episode described multiple ways: Keep highest priority ONLY

Examples:
a) Same disease, same temporal level: "diabetes, known diabetes, h/o diabetes" → Extract ONCE (E11.9)
b) Same disease, mixed temporal levels: "old stroke from 2020, prior CVA, history of cerebral infarction" → Keep Sequela (I69.398), all refer to same 2020 stroke
c) Different episodes, different times: "old stroke 2020, new acute stroke today" → Keep BOTH (I69.398 + I63.9)

When unsure if same episode: Look for dates, timeline clues, or explicit "old vs new" language.

Markers:
Sequela: history of, prior, old [disease], status post
Chronic: chronic, longstanding, established, known
Acute: acute, new onset, current, presenting with

5. SPECIFICITY VALIDATION
If code requires details NOT documented:
Laterality, stage, type, grade, deficit, complication
Downgrade to "unspecified" or "other" subtype

Never infer undocumented details.

6. CODE ASSIGNMENT
Select most specific BILLABLE code supported by evidence
Extract verbatim evidence (5-20 words)
Mark [CHRONIC] or [ACUTE]
Return JSON

CRITICAL RULES:

STATUS CODES (Z-codes): Include ONLY if:
Explicitly documented AND clinically relevant
Examples: stents (Z95.5), dialysis (Z99.2), transplant (Z94.*)

BILLABLE CODES ONLY:
If parent code selected: Choose appropriate child
If subtype unknown: Use "unspecified" child

CHRONIC MARKERS: history of, h/o, prior, chronic, longstanding, status post
These indicate CONFIRMED chronic diagnoses

Clinical Text:
{chunk}

Output JSON:
{{
  "diagnoses": [
    {{
      "condition": "[CHRONIC] Type 2 diabetes mellitus without complications",
      "icd10": "E11.9",
      "evidence_snippet": "history of type 2 diabetes"
    }}
  ]
}}

Rules:
Use valid 2026 ICD-10-CM codes only
ONE code per diagnosis (do NOT combine codes)
Extract BOTH chronic and acute conditions
Mark with "[CHRONIC]" or "[ACUTE]" prefix
EXCLUDE: negated, ruled-out, suspected, family history, symptoms when disease present
If no confirmed conditions, return empty list
Output STRICT JSON only
"""
)

ICD_SEMANTIC_BATCH_PROMPT = PromptTemplate(
    input_variables=["chunks_text", "num_chunks"],
    template="""
You are a medical coder using 2026 ICD-10-CM codes.

Extract CONFIRMED diagnoses from {num_chunks} clinical chunks. Process each chunk independently.

MANDATORY DECISION ALGORITHM (Apply to EACH chunk):

1. CERTAINTY FILTER
EXCLUDE if statement contains:
Uncertainty: possible, may, suspected, likely, cannot exclude, uncertain
Differential: versus, vs, rule out, R/O, concerning for, question of
Negation: no, denies, absent, negative for

INCLUDE if: confirmed, history of, diagnosed with, known, established

2. ENTITY TYPE FILTER
INCLUDE:
Named diseases (diabetes, pneumonia, stroke, fracture, infarct)
Disease nomenclature (-itis, -osis, -oma, -pathy)
Sequela with disease term (old infarct, prior MI, history of stroke)

EXCLUDE:
Descriptors without disease name:
  Structural: changes, atrophy, calcification, prominence, volume loss
  Process: small vessel disease, microvascular changes, ischemic changes
  Imaging: signal abnormality, degenerative changes, chronic changes
Pattern: [adjective] + [structure] without disease entity

3. SYMPTOM VS DISEASE
If symptom + underlying disease present: Code disease ONLY
If symptom alone: Code symptom
Symptoms: pain, fever, cough, nausea, vertigo, dizziness, fatigue
Check for causal relationship (symptom "due to/from" disease)

4. TEMPORAL PRIORITY (Within Chunk)
If same disease mentioned multiple times in this chunk: Keep highest priority
Priority: Sequela > Chronic > Acute

Different episodes of same disease type: Keep ALL
Same episode described multiple ways: Keep highest priority ONLY

Examples:
a) Same disease, same temporal level: "diabetes, known diabetes, h/o diabetes" → Extract ONCE (E11.9)
b) Same disease, mixed temporal levels: "old stroke from 2020, prior CVA, history of cerebral infarction" → Keep Sequela (I69.398), all refer to same 2020 stroke
c) Different episodes, different times: "old stroke 2020, new acute stroke today" → Keep BOTH (I69.398 + I63.9)

When unsure if same episode: Look for dates, timeline clues, or explicit "old vs new" language.

Markers:
Sequela: history of, prior, old [disease], status post
Chronic: chronic, longstanding, established, known
Acute: acute, new onset, current, presenting with

5. SPECIFICITY VALIDATION
If code requires details NOT documented:
Laterality, stage, type, grade, deficit, complication
Downgrade to "unspecified" or "other" subtype

Never infer undocumented details.

6. CODE ASSIGNMENT
Select most specific BILLABLE code supported by evidence
Extract verbatim evidence (5-20 words)
Mark [CHRONIC] or [ACUTE]
Return JSON

CRITICAL RULES:

STATUS CODES (Z-codes): Include ONLY if:
Explicitly documented AND clinically relevant
Examples: stents (Z95.5), dialysis (Z99.2), transplant (Z94.*)

BILLABLE CODES ONLY:
If parent code selected: Choose appropriate child
If subtype unknown: Use "unspecified" child

CHRONIC MARKERS: history of, h/o, prior, chronic, longstanding, status post
These indicate CONFIRMED chronic diagnoses

{chunks_text}

Output JSON (one result per chunk):
{{
  "results": [
    {{
      "chunk_number": 1,
      "diagnoses": [
        {{
          "condition": "[CHRONIC] Type 2 diabetes mellitus without complications",
          "icd10": "E11.9",
          "evidence_snippet": "history of diabetes"
        }}
      ]
    }}
  ]
}}

Rules:
Return exactly {num_chunks} results
Mark [CHRONIC] or [ACUTE]
ONE code per diagnosis
EXCLUDE: negated, ruled-out, suspected, family history, symptoms when disease present
If no confirmed conditions in chunk, return empty diagnoses list
Output STRICT JSON only
"""
)

ICD_GLOBAL_RECONCILIATION_PROMPT = PromptTemplate(
    input_variables=["all_diagnoses_text", "full_context_summary"],
    template="""You are a medical coder using 2026 ICD-10-CM codes.

You previously extracted diagnoses. Now reconcile them by applying these filters:

RECONCILIATION FILTERS (Apply in order):

1. UNCERTAINTY REMOVAL
Remove diagnoses with uncertainty in original evidence
Check: possible, cannot exclude, suspected, rule out

2. DESCRIPTOR REMOVAL
Remove imaging descriptors without disease names
Examples: small vessel disease, white matter changes, atrophy, calcifications

3. SYMPTOM REMOVAL
Remove symptoms if underlying disease exists
Keep symptoms only if no disease present

4. TEMPORAL RESOLUTION (Across All Chunks)
If same disease in multiple states: Keep Sequela > Chronic > Acute
Remove lower priority duplicates

Different episodes: Keep ALL (old MI + new MI = both coded)
Same episode, multiple terms: Keep highest priority ONLY

Examples:
a) Different episodes: "old MI 2019" (I25.2) + "acute MI today" (I21.9) → Keep BOTH
b) Same episode, mixed terms: "old MI 2019" (I25.2) + "chronic CAD from 2019 MI" (I25.10) → Keep I25.2 (Sequela > Chronic)
c) LLM confusion: "acute stroke 2020" (I63.9) + "prior stroke 2020" (I69.398) → Keep I69.398 (Sequela > Acute, correct for past event)

Use full document context to determine if different episodes or same episode.
Check for: dates, timeline clues, "old vs new" language, anatomical differences.
When ambiguous: default to same episode (conservative coding).

5. SUBTYPE DOWNGRADE
If code requires undocumented details: Downgrade to "unspecified"

6. MERGE DUPLICATES
Combine same condition from multiple chunks
Use most specific code

7. BILLABLE CODES ONLY
Replace parent codes with billable children

Previously Extracted:
{all_diagnoses_text}

Full Context:
{full_context_summary}

Output JSON:
{{
  "reconciled_diagnoses": [
    {{
      "condition": "[CHRONIC] Type 2 diabetes mellitus without complications",
      "icd10": "E11.9",
      "evidence_snippet": "history of diabetes",
      "source_chunks": [1, 3],
      "reasoning": "Merged from chunks 1,3; chronic diagnosis confirmed"
    }}
  ]
}}

Rules:
Apply filters sequentially
Remove duplicates (keep most specific)
If acute + chronic: Keep chronic
ONE entry per unique condition
Mark [CHRONIC] or [ACUTE]
Return STRICT JSON only
"""
)

GEM_SELECTION_PROMPT = PromptTemplate(
    input_variables=["icd9_code", "icd9_description", "icd10_candidates", "clinical_evidence", "clinical_context"],
    template="""You are a medical coder specializing in ICD code mapping.

Task: Select the SINGLE MOST APPROPRIATE ICD-10 code for the given ICD-9 code based on clinical context AND provide detailed reasoning.

ICD-9 Code: {icd9_code}
ICD-9 Description: {icd9_description}

Available ICD-10 Code Options (from GEM mapping):
{icd10_candidates}

Clinical Evidence (Direct Quote from Note):
{clinical_evidence}

Full Clinical Context from Patient Note:
{clinical_context}

Instructions:
1. Read the clinical evidence and context carefully
2. Consider the specific details mentioned (laterality, severity, complications, etc.)
3. Select the ICD-10 code that BEST matches the clinical documentation
4. Choose the most SPECIFIC code that is supported by the documentation
5. Prioritize evidence from the clinical evidence snippet, use full context for additional details
6. Provide detailed reasoning explaining WHY you selected this specific code

Return your response in the following JSON format:
{{
    "selected_code": "the ICD-10 code (e.g., B768)",
    "reasoning": "Detailed explanation of why this code was selected over others. Mention: (1) Why the ICD-9 code was initially extracted, (2) Why multiple ICD-10 mappings exist in GEM, (3) Which clinical evidence supports this specific code, (4) What makes this code more specific/accurate than the alternatives"
}}

The reasoning MUST be comprehensive and include:
- Why the original ICD-9 code was present
- Which specific clinical details led to choosing this particular ICD-10 code
- What makes this code more appropriate than the other candidates
"""
)

FAISS_CORRECTION_PROMPT = PromptTemplate(
    input_variables=["invalid_code", "condition", "candidates", "evidence_snippet"],
    template="""You are a certified medical coder specializing in ICD-10-CM coding.

CRITICAL RULES:
1. You MUST select a BILLABLE ICD-10-CM code (ALL candidates provided are billable)
2. You MUST NOT change the clinical meaning or severity
3. You MUST NOT map acute codes to sequela codes or vice versa
4. You MUST NOT downgrade specificity level

The extracted ICD-10 code '{invalid_code}' is INVALID.
The clinical condition is: '{condition}'

Clinical Evidence from Document:
{evidence_snippet}

Here are the top 5 most similar VALID and BILLABLE ICD-10 codes (from FAISS semantic search):
{candidates}

Task:
Select the SINGLE ICD-10 code that BEST matches the condition AND provide detailed reasoning.

Selection Criteria (in priority order):
1. CLINICAL ACCURACY - Must match the exact clinical meaning
2. SPECIFICITY - Choose the most specific code that matches the documentation
3. SEVERITY - Must preserve documented severity level (mild, moderate, severe, class I/II/III)
4. LATERALITY - Must preserve side specification (left, right, bilateral, unilateral)
5. EPISODE - Must preserve temporal classification (acute, chronic, sequela, initial/subsequent encounter)
6. ANATOMICAL SITE - Must preserve specific body location if documented
7. TYPE/SUBTYPE - Must preserve disease type, complication type, or subclassification

FORBIDDEN Actions:
❌ DO NOT map to parent/non-specific codes
❌ DO NOT change temporal classification (acute ↔ chronic ↔ sequela)
❌ DO NOT change disease category or system
❌ DO NOT downgrade severity or specificity
❌ DO NOT change laterality or anatomical location
❌ DO NOT substitute "unspecified" codes when specific codes match

Return your response in the following JSON format:
{{
    "selected_code": "the ICD-10 code (e.g., E119)",
    "reasoning": "Detailed explanation of the correction. Mention: (1) Why the original code '{invalid_code}' was invalid (wrong format, non-existent, etc.), (2) How FAISS semantic search found similar codes, (3) Which specific clinical details from the evidence support this code, (4) Why this code is the most accurate among the top 5 candidates, (5) What makes this code preserve the clinical meaning and specificity"
}}

The reasoning MUST be comprehensive and include:
- Why the original code was invalid
- How semantic search helped find the correction
- Which clinical evidence supports the selected code
- Why this specific code is the best match among candidates
"""
)
