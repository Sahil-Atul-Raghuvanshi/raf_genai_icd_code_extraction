# Why H90.9 Is Not Being Corrected - Investigation

## Issue

`H90.9` appears in `invalid_semantic_icd_codes` but does NOT appear in `corrected_codes` or `correction_details`.

## What to Check

Run the app again and look for this in the console:

```
🔍 Chunk 4: Found 2 invalid semantic codes: ['2022', 'H90.9']
   Conditions mapped: 2 codes
   - 2022: 'some condition...'
   - H90.9: 'Bilateral sensorineural hearing loss...'

📊 Smart Correction Filtering:
  Total invalid codes: 2
  ⚡ Instant fixes: 0
  🤖 Needs LLM: 1  ← Should see H90.9 here
  ⏭️  Skipped: 1
  
  Will correct with LLM:
    - H90.9 ('Bilateral sensorineural hearing loss...')  ← Confirm it's here
    
  Skipped codes:
    - 2022: low_confidence_0.20  ← This is expected
```

## Possible Causes

### Cause 1: Low Confidence (Most Likely)
The condition text for `H90.9` has a confidence score below 0.4.

**Why this happens:**
- Condition text is too short (< 15 characters)
- Missing specific medical details
- Contains vague terms like "unspecified"

**Example:**
```python
# LOW confidence (< 0.4) - Will be SKIPPED
condition = "hearing loss"  # Too short, too vague
confidence = 0.3

# HIGH confidence (> 0.4) - Will be CORRECTED
condition = "Bilateral sensorineural hearing loss, moderate"
confidence = 0.7
```

### Cause 2: No Condition Mapped
The code has no associated condition text from the diagnosis objects.

**Check console output:**
```
- H90.9: 'NO CONDITION'  ← If you see this, that's the problem
```

### Cause 3: Format Check Fails
The code doesn't match valid ICD format patterns.

**But this is unlikely** since `H90.9` is a valid ICD format.

## Solutions

### Solution 1: Lower Confidence Threshold (Quick Fix)

**In `app.py` line 258:**
```python
# Current:
confidence_threshold=0.4,  # Only correct if confidence > 40%

# Change to:
confidence_threshold=0.2,  # More lenient, will correct H90.9
```

This will allow corrections for codes with lower confidence conditions.

### Solution 2: Improve Condition Extraction (Better Fix)

The real issue is that the LLM is not extracting detailed enough condition text for `H90.9`.

**Update the prompt** in `clinical_extraction/prompts.py` to emphasize:
- Writing detailed conditions
- Including specific medical terminology
- Following ICD-10 long description format

### Solution 3: Override for Specific Codes (Targeted Fix)

Add special handling for codes that look valid:

**In `icd_mapping/correction_filter.py`, update `should_correct_code()` around line 187:**

```python
# Check 2: Low confidence condition?
confidence = calculate_condition_confidence(condition_text, evidence_snippet)

# Override: If code looks like a valid ICD code format, reduce threshold
if re.match(r'^[A-Z]\d{2}\.\d{1,2}$', invalid_code):
    # This is a proper ICD-10 format (e.g., H90.9)
    # Use lower threshold since it's likely just mis-validated
    effective_threshold = min(confidence_threshold, 0.2)
else:
    effective_threshold = confidence_threshold

if confidence < effective_threshold:
    return False, None, f"low_confidence_{confidence:.2f}"
```

## What the Debug Output Will Tell You

### Scenario A: H90.9 is Being Skipped
```
Skipped codes:
  - H90.9: low_confidence_0.35  ← Confidence is 0.35, below 0.4 threshold
```

**Fix**: Lower the confidence threshold to 0.2

### Scenario B: H90.9 Has No Condition
```
🔍 Chunk 4: Found 2 invalid semantic codes: ['2022', 'H90.9']
   - H90.9: 'NO CONDITION'  ← Problem here!
```

**Fix**: There's an issue with the diagnosis object mapping. The condition is not being preserved.

### Scenario C: H90.9 Is Being Corrected
```
Will correct with LLM:
  - H90.9 ('Bilateral sensorineural hearing loss')  ← It's here!
  
🤖 LLM corrections: 1
📋 Total corrected codes added: 1  ← But this should be > 0
```

**Fix**: The LLM correction itself is failing. Check for FAISS or LLM API errors.

## Recommended Action

1. **Run the app** and **check the console output** carefully
2. **Look for `H90.9`** specifically in the debug output
3. **Share the exact console output** for the chunk containing `H90.9`

This will tell us exactly why it's not being corrected:
- Is it being skipped due to low confidence?
- Is there no condition mapped to it?
- Is the LLM correction failing?
- Something else?

## Quick Test

To force correction of all codes including `H90.9`, temporarily change:

**In `app.py` line 258:**
```python
confidence_threshold=0.0,  # Force correction of ALL codes
```

Then run again. If `H90.9` now appears in `corrected_codes`, you know it was a confidence issue.
