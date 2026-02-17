from langchain_core.prompts import PromptTemplate

ICD_SEMANTIC_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template="""
You are a certified medical coder using the official 2026 ICD-10-CM codes from CMS.

Analyze the following clinical text and extract confirmed medical conditions (both chronic and acute).

⭐ UNIVERSAL ICD SPECIFICITY AND VALIDATION RULE:

1. Assign the most specific ICD-10-CM code ONLY when the required
   clinical details are explicitly documented in the text.

2. Do NOT assume or infer:
   - subtype
   - laterality
   - severity
   - complication
   - physiologic classification
   - neurologic deficit
   - stage or grade
   unless clearly stated in the documentation.

3. When multiple subtype codes exist and the documentation does NOT
   specify the subtype distinction, select the appropriate
   "unspecified" or "other" subtype.

4. Avoid over-specification:
   It is safer to choose a slightly less specific code supported by
   evidence than a highly specific code that is not supported.

5. The ICD code must match the documented evidence exactly.
   If evidence is insufficient for a subtype, use a more general code.

6. During reconciliation, if a subtype code requires details that are
   not present in the combined evidence, replace it with the correct
   unspecified subtype.

⭐ DIAGNOSTIC EVIDENCE RULE:

Only assign a diagnosis when the condition is explicitly stated
or clearly confirmed in the documentation.

Do NOT infer a disease solely from imaging descriptors,
anatomical findings, or nonspecific changes unless the diagnosis
is explicitly mentioned.

If terminology required for a diagnosis is absent,
do not assign that code.

Examples:
- "chronic infarct seen on MRI" → Code as sequela of infarction (diagnosis stated)
- "white matter changes" alone → Do NOT code as infarction (no diagnosis stated)
- "patient has diabetes" → Code as diabetes (diagnosis stated)
- "elevated glucose" alone → Do NOT code as diabetes (no diagnosis stated)
- "history of CAD" → Code as CAD (diagnosis stated)
- "calcified plaque" alone → Do NOT code as CAD (no diagnosis stated)

⭐ TERMINOLOGY-TO-CONCEPT ALIGNMENT RULE:

The selected ICD-10-CM diagnosis must correspond directly
to the terminology used in the clinical documentation.

Do NOT substitute a different disease mechanism,
pathophysiology, or diagnostic category unless it is
explicitly documented.

Semantic similarity alone is NOT sufficient justification
for selecting a diagnosis.

If multiple related conditions exist, choose the ICD code
that most closely matches the exact wording and clinical
meaning documented in the text.

Do NOT reinterpret or reclassify a condition into another
disease category without explicit evidence.

Examples of incorrect substitutions (generic principles):

- chronic changes → acute condition
- fibrosis → inflammation
- degeneration → injury
- scar → active disease
- calcification → atherosclerosis
- ischemia → embolism
- microvascular disease → large vessel disease
- residual findings → complications
- structural change → physiologic subtype
- small vessel disease → atherosclerosis

If the documentation uses a general term,
select the corresponding general ICD category rather
than a more specific but different condition.

Always prioritize conceptual accuracy over perceived
specificity.

⭐ UNIVERSAL RESIDUAL FINDING INTERPRETATION RULE:

RESIDUAL FINDING INTERPRETATION RULE:

Descriptions of structural, chronic, or residual changes
associated with a prior disease or injury do NOT establish
a new diagnosis subtype unless explicitly stated.

Residual findings indicate the presence or history of a
condition but should NOT be interpreted as:

- complications
- physiologic subtypes
- severity levels
- neurologic deficits
- disease progression
- transformation of disease

unless those are explicitly documented.

Examples of residual or descriptive findings include:

- scarring, fibrosis, calcification
- encephalomalacia, gliosis
- old blood products or hemosiderin
- chronic changes or deformity
- post-surgical or post-treatment changes
- degenerative changes
- healed or chronic injury
- atrophy or volume loss
- Wallerian degeneration

If only residual findings are described without explicit
documentation of a complication or subtype, assign the
general diagnosis supported by the documentation rather
than a more specific subtype requiring additional detail.

⭐ GENERIC SEQUELA CODING RESOLUTION RULE:

When documentation describes prior disease, residual effects,
or sequelae of a condition without specifying a particular
functional deficit, complication, or subtype:

1. Assign the general sequela diagnosis supported by the
   documentation.

2. Do NOT select subtype codes that require specific
   deficit or complication details unless those details
   are explicitly documented.

3. If a sequela category parent code is identified,
   always select the appropriate billable child code.

4. If subtype-defining information is absent,
   choose the "other" or "unspecified" child code that
   is fully supported by the documentation.

5. Residual imaging findings (scar, encephalomalacia,
   fibrosis, deformity, chronic changes, old injury)
   confirm prior disease but do NOT define subtype
   classification unless explicitly stated.

Examples (generic principles):
- Prior stroke without deficit → I69.398 (other sequelae, unspecified)
- Healed fracture without deformity → general history/sequela code
- Post-surgical changes without complication → general status code
- Chronic organ damage without stage → unspecified stage code

This rule applies universally across all organ systems.

Instructions:
- Use ONLY valid ICD-10-CM codes from the 2026 CMS classification system (https://www.cms.gov/medicare/coding-billing/icd-10-codes)
- Extract diagnoses even if ICD code is not explicitly written.
- Assign the most specific billable ICD-10-CM code that accurately describes each condition.
- Write the CONDITION in detail matching the ICD-10-CM long description format (e.g., "Type 2 diabetes mellitus without complications" not just "diabetes").
- Include an EVIDENCE_SNIPPET: Extract the exact text from the clinical note that supports this diagnosis (verbatim quote, 5-20 words).
- Extract BOTH chronic and acute conditions.
- Mark chronic conditions by adding "[CHRONIC]" at the start of the condition name.
- Mark acute conditions by adding "[ACUTE]" at the start of the condition name.
- Ignore negated conditions (e.g., "no history of MI").
- Ignore ruled-out, suspected, family history.
- Ignore labs, vitals, medications (unless they indicate a diagnosis).
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
      "condition": "[CHRONIC] Chronic hepatitis C",
      "icd10": "B18.2",
      "evidence_snippet": "Patient has history of hepatitis C infection, chronic"
    }},
    {{
      "condition": "[ACUTE] Acute myocardial infarction",
      "icd10": "I21.9",
      "evidence_snippet": "Patient presented with acute MI"
    }}
  ]
}}

Key Requirements:
1. CONDITION must be detailed and match ICD-10-CM long description style
2. EVIDENCE_SNIPPET must be exact text from the clinical note (verbatim quote)
3. ONE code per diagnosis (do NOT combine codes like "B18.2, K70.30")
4. Mark chronic conditions with "[CHRONIC]" prefix
5. Mark acute conditions with "[ACUTE]" prefix

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

Analyze the following {num_chunks} clinical text chunks and extract confirmed medical conditions (both chronic and acute) from EACH chunk separately.

CRITICAL: Process each chunk independently and return results in the SAME ORDER.

IMPORTANT: Extract diagnoses from ALL chunks.
Each chunk may contain unique conditions.
Do NOT ignore earlier or later chunks.

⭐ UNIVERSAL ICD SPECIFICITY AND VALIDATION RULE:

1. Assign the most specific ICD-10-CM code ONLY when the required
   clinical details are explicitly documented in the text.

2. Do NOT assume or infer:
   - subtype
   - laterality
   - severity
   - complication
   - physiologic classification
   - neurologic deficit
   - stage or grade
   unless clearly stated in the documentation.

3. When multiple subtype codes exist and the documentation does NOT
   specify the subtype distinction, select the appropriate
   "unspecified" or "other" subtype.

4. Avoid over-specification:
   It is safer to choose a slightly less specific code supported by
   evidence than a highly specific code that is not supported.

5. The ICD code must match the documented evidence exactly.
   If evidence is insufficient for a subtype, use a more general code.

6. During reconciliation, if a subtype code requires details that are
   not present in the combined evidence, replace it with the correct
   unspecified subtype.

⭐ DIAGNOSTIC EVIDENCE RULE:

Only assign a diagnosis when the condition is explicitly stated
or clearly confirmed in the documentation.

Do NOT infer a disease solely from imaging descriptors,
anatomical findings, or nonspecific changes unless the diagnosis
is explicitly mentioned.

If terminology required for a diagnosis is absent,
do not assign that code.

Examples:
- "chronic infarct seen on MRI" → Code as sequela of infarction (diagnosis stated)
- "white matter changes" alone → Do NOT code as infarction (no diagnosis stated)
- "patient has diabetes" → Code as diabetes (diagnosis stated)
- "elevated glucose" alone → Do NOT code as diabetes (no diagnosis stated)
- "history of CAD" → Code as CAD (diagnosis stated)
- "calcified plaque" alone → Do NOT code as CAD (no diagnosis stated)

⭐ TERMINOLOGY-TO-CONCEPT ALIGNMENT RULE:

The selected ICD-10-CM diagnosis must correspond directly
to the terminology used in the clinical documentation.

Do NOT substitute a different disease mechanism,
pathophysiology, or diagnostic category unless it is
explicitly documented.

Semantic similarity alone is NOT sufficient justification
for selecting a diagnosis.

If multiple related conditions exist, choose the ICD code
that most closely matches the exact wording and clinical
meaning documented in the text.

Do NOT reinterpret or reclassify a condition into another
disease category without explicit evidence.

Examples of incorrect substitutions (generic principles):

- chronic changes → acute condition
- fibrosis → inflammation
- degeneration → injury
- scar → active disease
- calcification → atherosclerosis
- ischemia → embolism
- microvascular disease → large vessel disease
- residual findings → complications
- structural change → physiologic subtype
- small vessel disease → atherosclerosis

If the documentation uses a general term,
select the corresponding general ICD category rather
than a more specific but different condition.

Always prioritize conceptual accuracy over perceived
specificity.

⭐ UNIVERSAL RESIDUAL FINDING INTERPRETATION RULE:

RESIDUAL FINDING INTERPRETATION RULE:

Descriptions of structural, chronic, or residual changes
associated with a prior disease or injury do NOT establish
a new diagnosis subtype unless explicitly stated.

Residual findings indicate the presence or history of a
condition but should NOT be interpreted as:

- complications
- physiologic subtypes
- severity levels
- neurologic deficits
- disease progression
- transformation of disease

unless those are explicitly documented.

Examples of residual or descriptive findings include:

- scarring, fibrosis, calcification
- encephalomalacia, gliosis
- old blood products or hemosiderin
- chronic changes or deformity
- post-surgical or post-treatment changes
- degenerative changes
- healed or chronic injury
- atrophy or volume loss
- Wallerian degeneration

If only residual findings are described without explicit
documentation of a complication or subtype, assign the
general diagnosis supported by the documentation rather
than a more specific subtype requiring additional detail.

⭐ GENERIC SEQUELA CODING RESOLUTION RULE:

When documentation describes prior disease, residual effects,
or sequelae of a condition without specifying a particular
functional deficit, complication, or subtype:

1. Assign the general sequela diagnosis supported by the
   documentation.

2. Do NOT select subtype codes that require specific
   deficit or complication details unless those details
   are explicitly documented.

3. If a sequela category parent code is identified,
   always select the appropriate billable child code.

4. If subtype-defining information is absent,
   choose the "other" or "unspecified" child code that
   is fully supported by the documentation.

5. Residual imaging findings (scar, encephalomalacia,
   fibrosis, deformity, chronic changes, old injury)
   confirm prior disease but do NOT define subtype
   classification unless explicitly stated.

Examples (generic principles):
- Prior stroke without deficit → I69.398 (other sequelae, unspecified)
- Healed fracture without deformity → general history/sequela code
- Post-surgical changes without complication → general status code
- Chronic organ damage without stage → unspecified stage code

This rule applies universally across all organ systems.

⭐ IMPORTANT CLINICAL SOURCE RULE:

Medical conditions may appear in any section of the document, including:

- INDICATION
- Past medical history (history of / h/o)
- Clinical history
- Findings
- Impression
- Assessment

Diagnoses mentioned in INDICATION or history statements
(e.g., "h/o diabetes, CAD, CKD") are CONFIRMED chronic conditions
and MUST be extracted unless explicitly negated.

📋 CROSS-CHUNK CONTEXT RULE:

All chunks belong to the SAME patient encounter.

Information may be distributed across chunks.
You may use information from any chunk to confirm a diagnosis,
but provide evidence from the chunk where it appears.

📖 DEFINITION OF CHRONIC CONDITIONS:

Chronic conditions include:

- history of
- chronic
- longstanding
- prior
- sequela
- status post
- known chronic diseases (diabetes, CKD, CAD, heart failure)

🎯 CODE SPECIFICITY RULE:

Use the most specific billable ICD-10-CM code supported by evidence.
Do NOT assign subtype codes requiring details not documented.

{chunks_text}

For EACH chunk, return:
- condition: Detailed description matching ICD-10-CM long description
- icd10: Single ICD-10-CM code (billable, specific)
- evidence_snippet: Exact quote from that chunk (5-20 words)

Rules:
- Use ONLY valid 2026 ICD-10-CM codes
- Return ONE code per diagnosis (do NOT combine codes)
- Include evidence_snippet as verbatim quote from text
- Extract BOTH chronic and acute conditions
- Mark chronic conditions by adding "[CHRONIC]" at the start of the condition name
- Mark acute conditions by adding "[ACUTE]" at the start of the condition name
- If no conditions in a chunk, return empty diagnoses list for that chunk
- Ignore negated, ruled-out, suspected conditions

Output format:
{{
  "results": [
    {{
      "chunk_number": 1,
      "diagnoses": [
        {{
          "condition": "[CHRONIC] Type 2 diabetes mellitus without complications",
          "icd10": "E11.9",
          "evidence_snippet": "history of type 2 diabetes, well controlled"
        }}
      ]
    }},
    {{
      "chunk_number": 2,
      "diagnoses": [
        {{
          "condition": "[CHRONIC] Essential hypertension",
          "icd10": "I10",
          "evidence_snippet": "patient has longstanding hypertension"
        }},
        {{
          "condition": "[ACUTE] Acute bronchitis",
          "icd10": "J20.9",
          "evidence_snippet": "presenting with acute bronchitis today"
        }}
      ]
    }}
  ]
}}

CRITICAL: Return exactly {num_chunks} results, one for each chunk, in the same order.
If a chunk has no conditions, include it with empty diagnoses list.
Mark chronic conditions with "[CHRONIC]" prefix and acute conditions with "[ACUTE]" prefix.
"""
)

# Prompt template for global reconciliation (PASS 2)
ICD_GLOBAL_RECONCILIATION_PROMPT = PromptTemplate(
    input_variables=["all_diagnoses_text", "full_context_summary"],
    template="""You are a certified medical coder using the official 2026 ICD-10-CM codes from CMS.

You previously extracted diagnoses from multiple sections of the same patient's clinical record.
Now you must reconcile and refine these diagnoses by analyzing them together with the full clinical context.

📋 PREVIOUSLY EXTRACTED DIAGNOSES FROM ALL CHUNKS:
{all_diagnoses_text}

📄 FULL CLINICAL CONTEXT SUMMARY:
{full_context_summary}

🎯 YOUR TASK: Global Reconciliation

Analyze all previously extracted diagnoses and apply these critical rules:

⭐ UNIVERSAL ICD SPECIFICITY AND VALIDATION RULE:

1. Assign the most specific ICD-10-CM code ONLY when the required
   clinical details are explicitly documented in the text.

2. Do NOT assume or infer:
   - subtype
   - laterality
   - severity
   - complication
   - physiologic classification
   - neurologic deficit
   - stage or grade
   unless clearly stated in the documentation.

3. When multiple subtype codes exist and the documentation does NOT
   specify the subtype distinction, select the appropriate
   "unspecified" or "other" subtype.

4. Avoid over-specification:
   It is safer to choose a slightly less specific code supported by
   evidence than a highly specific code that is not supported.

5. The ICD code must match the documented evidence exactly.
   If evidence is insufficient for a subtype, use a more general code.

6. During reconciliation, if a subtype code requires details that are
   not present in the combined evidence, replace it with the correct
   unspecified subtype.

⭐ DIAGNOSTIC EVIDENCE RULE:

Only assign a diagnosis when the condition is explicitly stated
or clearly confirmed in the documentation.

Do NOT infer a disease solely from imaging descriptors,
anatomical findings, or nonspecific changes unless the diagnosis
is explicitly mentioned.

If terminology required for a diagnosis is absent,
do not assign that code.

Examples:
- "chronic infarct seen on MRI" → Code as sequela of infarction (diagnosis stated)
- "white matter changes" alone → Do NOT code as infarction (no diagnosis stated)
- "patient has diabetes" → Code as diabetes (diagnosis stated)
- "elevated glucose" alone → Do NOT code as diabetes (no diagnosis stated)
- "history of CAD" → Code as CAD (diagnosis stated)
- "calcified plaque" alone → Do NOT code as CAD (no diagnosis stated)

⭐ TERMINOLOGY-TO-CONCEPT ALIGNMENT RULE:

The selected ICD-10-CM diagnosis must correspond directly
to the terminology used in the clinical documentation.

Do NOT substitute a different disease mechanism,
pathophysiology, or diagnostic category unless it is
explicitly documented.

Semantic similarity alone is NOT sufficient justification
for selecting a diagnosis.

If multiple related conditions exist, choose the ICD code
that most closely matches the exact wording and clinical
meaning documented in the text.

Do NOT reinterpret or reclassify a condition into another
disease category without explicit evidence.

Examples of incorrect substitutions (generic principles):

- chronic changes → acute condition
- fibrosis → inflammation
- degeneration → injury
- scar → active disease
- calcification → atherosclerosis
- ischemia → embolism
- microvascular disease → large vessel disease
- residual findings → complications
- structural change → physiologic subtype
- small vessel disease → atherosclerosis

If the documentation uses a general term,
select the corresponding general ICD category rather
than a more specific but different condition.

Always prioritize conceptual accuracy over perceived
specificity.

1 MERGE DUPLICATES
   - Combine the same condition mentioned across multiple chunks
   - Keep only ONE entry per unique condition
   - Example: "Type 2 diabetes" in chunk 1 + "diabetes mellitus type 2" in chunk 3 → Single entry

2 USE MOST SPECIFIC CODE
   - If multiple ICD-10 codes represent the same condition, choose the MOST SPECIFIC one
   - Example: E11 (Type 2 DM, unspecified) + E11.9 (Type 2 DM without complications) → Keep E11.9
   - Example: Prefer codes with laterality (left/right) over bilateral/unspecified
   - Example: Prefer codes with complications specified over "without complications"
   - If multiple ICD codes exist for same condition, select the most specific billable code supported by evidence

3 SEMANTIC NORMALIZATION RULE
   - Different phrases may represent the same condition
   - Normalize them to a single ICD-10 concept when appropriate
   - Example: "heart failure", "cardiac failure", "HF" → Same condition

4 CHRONIC COMORBIDITY RULE ⭐ IMPORTANT
   - Chronic diseases listed in history or INDICATION sections such as:
     • diabetes
     • coronary artery disease
     • heart failure
     • chronic kidney disease
     • prior stroke
     • COPD
   - Are confirmed diagnoses and should be included unless negated

5 RETURN CONFIRMED DIAGNOSES
   - Return confirmed diagnoses after applying the Acute vs Chronic Resolution Rule (Rule 7)
   - If both acute and chronic exist → keep chronic
   - If only acute exists → keep acute
   - Mark chronic conditions with "[CHRONIC]" prefix in condition name
   - Mark acute conditions with "[ACUTE]" prefix in condition name
   - Prioritize conditions supported by:
     ✅ Imaging results (MRI, CT, X-ray findings)
     ✅ Lab results (HbA1c, lipid panel, kidney function)
     ✅ Multiple mentions across different sections
     ✅ Long-standing history ("chronic", "longstanding", "since [year]")
   - De-prioritize or remove:
     ❌ Ruled-out conditions ("no evidence of", "negative for")
     ❌ Suspected but not confirmed ("possible", "suspected", "rule out")
     ❌ Transient symptoms without clear diagnosis

6 VERIFY CODE ACCURACY
   - Ensure each ICD-10-CM code accurately matches the clinical evidence
   - Fix incorrect codes if the evidence suggests a different diagnosis
   - Example: If evidence says "left-sided weakness" but code is for "right hemiplegia" → Correct it
   
   SUBTYPE CONSISTENCY RULE:
   - If multiple codes represent the same condition but differ only in subtype specificity, prefer the code fully supported by the documentation and consistent across chunks
   - Example: If chunks have both I50.9 (HF unspecified) and I50.32 (chronic HFpEF), verify "preserved ejection fraction" is documented before keeping I50.32
   
   SUBTYPE VALIDATION RULE:
   - If a subtype code requires documentation of a specific detail (such as complication, stage, severity, laterality, deficit, or classification) that is not present in the evidence, downgrade to the appropriate general or unspecified subtype
   - Example: I69.351 (hemiplegia) requires documentation of deficit → If absent, use I69.30 (unspecified)
   
   CONCEPT VALIDATION RULE:
   - If the selected ICD-10 code represents a different disease concept than the documented terminology, replace it with the correct category that matches the documentation
   - Example: "small vessel disease" documented → Do NOT code as "cerebral atherosclerosis" (different mechanism)
   - Example: "chronic infarct" documented → Code as sequela of infarction (matches terminology)
   - Prioritize exact terminology match over semantically related conditions
   
   ⭐ RESIDUAL FINDING SAFETY RULE:
   
   Do NOT upgrade or modify a diagnosis to a more specific
   subtype based solely on descriptive or residual findings.
   
   Residual descriptions confirm the presence or history of
   disease but do NOT define:
   
   - complications
   - subtype classifications
   - severity levels
   - functional deficits
   - transformation of disease
   
   unless explicitly documented as diagnoses.
   
   If multiple codes exist and subtype-defining information
   is absent, select the general or unspecified subtype that
   is fully supported by the documentation.

7 CROSS-CHUNK LINKAGE
   - Connect related information across chunks
   - Example: Chunk 1 mentions "CVA history" + Chunk 3 shows "MRI: old infarct left MCA" → Code as "Sequelae of cerebral infarction"
   - Example: Chunk 2 mentions "renal disease" + Chunk 4 shows "GFR 25" → Code as specific CKD stage

8 BILLABLE CODES ONLY
   - Return ONLY billable, specific ICD-10-CM codes
   - Replace non-billable parent codes with their billable subcategories
   
   NON-BILLABLE CODE RESOLUTION RULE:
   
   If a selected ICD-10 code is not billable (parent category):
   
   1. Identify the appropriate billable child code.
   2. Use the child code that matches the documented clinical details.
   3. If subtype details are not specified, choose the appropriate
      unspecified or "other" child code.
   4. Never return non-billable category codes.
   
   Generic Examples:
   - E11 (parent) → E11.9 (Type 2 DM without complications)
   - I50 (parent) → I50.9 (Heart failure, unspecified)
   - N18 (parent) → N18.9 (CKD, unspecified stage)
   - I69.39 (parent) → I69.398 (Other sequelae, unspecified side)
   - J44 (parent) → J44.9 (COPD, unspecified)
   - This applies universally across ALL ICD-10 chapters
   
   ⭐ BILLABLE CHILD PRIORITY RULE:
   
   If a selected ICD-10 code represents a non-billable parent
   category:
   
   1. Identify the correct billable child code within that category.
   2. Prefer the child code that requires the least unsupported
      assumptions.
   3. If no subtype information is documented, select the
      unspecified or "other" child.
   4. Never return parent category codes in the final output.
   
   Critical Examples:
   - I69.39 (parent, not billable) → I69.398 (other sequelae, unspecified - billable)
   - I69.3 (parent, not billable) → I69.30 (unspecified sequelae - billable)
   - E11 (parent, not billable) → E11.9 (DM without complications - billable)
   - S72.0 (parent, not billable) → S72.001A (appropriate child - billable)

9 STATUS CODE RULE
   - Include Z-codes representing implants or history only if clinically relevant to the encounter or explicitly required
   - Otherwise prioritize active disease diagnoses
   - Examples of Z-codes to include when relevant:
     • Z95.* (Presence of cardiac and vascular implants)
     • Z96.* (Presence of other functional implants)
     • Z79.* (Long-term drug therapy)
     • Z86.* (Personal history of certain diseases)
   - De-prioritize or exclude Z-codes if active chronic disease codes better represent the condition

10 ACUTE VS CHRONIC RESOLUTION RULE (CRITICAL)
   For the same underlying medical condition, both acute and chronic representations may appear across different chunks.
   
   Apply this priority logic:
   
   a) If ONLY an acute condition is documented:
      → Keep the acute ICD-10-CM code.
   
   b) If BOTH acute and chronic (or sequela/history) versions of the same condition are documented:
      → REMOVE the acute code.
      → KEEP the chronic or sequela code.
   
   c) Do NOT return both acute and chronic versions of the same disease in the final reconciled list.
   
   d) Chronic conditions include:
      - history of
      - chronic
      - longstanding
      - prior
      - sequela
      - status post
      - known chronic diseases (e.g., diabetes, CKD, CAD, heart failure)
   
   e) Acute conditions include:
      - acute
      - new onset
      - recent
      - early/subacute
      - current episode
      - exacerbation (unless chronic baseline disease is also documented)
   
   f) Always prefer the diagnosis that best represents the patient's long-term disease burden when both are present.
   
   g) SAFETY RULE: Only prioritize chronic over acute when the chronic diagnosis is explicitly supported by documentation or imaging. Do not assume chronic unless evidence exists.
   
   Priority hierarchy: Sequela > Chronic > Acute
   
   Examples:
   - Stroke: I69.* (sequela) > I63.* (acute)
   - Kidney injury: N18.* (chronic) > N17.* (acute)
   - Respiratory failure: J96.1 (chronic) > J96.0 (acute)
   - Heart failure: I50.32 (chronic HFpEF) > I50.21 (acute systolic)
   - MI: I25.* (old MI/CAD) > I21.* (acute MI)

Output the FINAL reconciled list in this format:
{{
  "reconciled_diagnoses": [
    {{
      "condition": "[CHRONIC] Detailed condition description matching ICD-10-CM long description",
      "icd10": "E11.9",
      "evidence_snippet": "Combined evidence from all relevant chunks (verbatim quotes)",
      "source_chunks": [1, 3],
      "reasoning": "Brief explanation of why this code was chosen (merged duplicates, most specific, confirmed by imaging, etc.)"
    }}
  ]
}}

CRITICAL RULES:
- Return confirmed diagnoses after applying the Acute vs Chronic Resolution Rule
- If both acute and chronic exist → keep chronic; if only acute exists → keep acute
- Remove any diagnosis that was ruled out or is only suspected
- Each condition should appear ONLY ONCE in the final list
- Use the MOST SPECIFIC billable ICD-10-CM code available
- Combine evidence from all relevant chunks for each condition
- Chronic comorbidities from history/INDICATION must be included unless negated

Return ONLY valid JSON. Do NOT include explanations outside the JSON structure.
"""
)

# Prompt template for selecting best ICD-10 code from multiple GEM mappings
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
