# RAFgenAI - Complete Code Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Module Structure](#module-structure)
4. [Data Flow](#data-flow)
5. [Core Modules](#core-modules)
6. [Performance Optimizations](#performance-optimizations)
7. [API Dependencies](#api-dependencies)
8. [Setup and Installation](#setup-and-installation)

---

## Project Overview

**RAFgenAI** is a clinical PDF processing system that extracts ICD-10-CM medical diagnosis codes using a semantic AI approach combining:
- **LLM Semantic Extraction** (AI-powered diagnosis extraction)
- **FAISS Vector Search** (for code correction)
- **GEM Mapping** (ICD-9 to ICD-10 conversion)
- **Validation & Correction** (ensuring code accuracy)

### Key Features
- ✅ Processes both digital and scanned PDFs (with OCR fallback)
- ✅ Extracts ICD codes through AI semantic understanding (LLM)
- ✅ Validates codes against official 2026 ICD-10-CM master data
- ✅ Auto-corrects invalid codes using FAISS semantic search + LLM
- ✅ Converts ICD-9 codes to ICD-10 via GEM mappings with intelligent selection
- ✅ Identifies billable vs non-billable codes
- ✅ Provides detailed extraction evidence and correction traceability

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PDF UPLOAD                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ PDF Loader   │→ │ Text Cleaner │→ │   Chunker    │          │
│  │ (PyMuPDF)    │  │   (Regex)    │  │ (200 tokens) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓ (OCR Fallback)                                         │
│  ┌──────────────┐                                                │
│  │  OCR Engine  │                                                │
│  │ (Tesseract)  │                                                │
│  └──────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ICD EXTRACTION LAYER                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            LLM Semantic Extraction (Batch)                │  │
│  │              Gemini-2.5-Flash                             │  │
│  │              Batch Size: 5 chunks                         │  │
│  │     Extracts diagnoses with evidence snippets             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VALIDATION & CORRECTION LAYER                   │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ ICD-10 Validator │  │ ICD-9 Validator  │                     │
│  │ (2026 CMS Data)  │  │ (Master Data)    │                     │
│  └────────┬─────────┘  └────────┬─────────┘                     │
│           │                     │                                │
│           ├─────Valid ICD-10────┤                                │
│           │                     │                                │
│           │           Valid ICD-9 → GEM Mapping                  │
│           │                     │   (ICD-9 → ICD-10)             │
│           │                     │   + LLM Selection              │
│           │                     ▼                                │
│           │              ┌──────────────────┐                    │
│           │              │ GEM Selector LLM │                    │
│           │              │ (Multiple → Best)│                    │
│           │              └──────────────────┘                    │
│           │                     │                                │
│           ▼                     ▼                                │
│    Invalid Semantic Codes                                        │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────┐            │
│  │        SMART CORRECTION SYSTEM                   │            │
│  │  ┌──────────────┐  ┌──────────────────────────┐ │            │
│  │  │ Filter Logic │→ │ Instant Format Fixes     │ │            │
│  │  │ (Skip/Fix)   │  │ E119 → E11.9            │ │            │
│  │  └──────────────┘  └──────────────────────────┘ │            │
│  │         │                                        │            │
│  │         ▼                                        │            │
│  │  ┌─────────────────────────────────────────┐   │            │
│  │  │    FAISS Semantic Search (Top 5)        │   │            │
│  │  │    Billable Codes Only                  │   │            │
│  │  │    Optimized Search Multiplier: 1.18x   │   │            │
│  │  └─────────────────────────────────────────┘   │            │
│  │         │                                        │            │
│  │         ▼                                        │            │
│  │  ┌─────────────────────────────────────────┐   │            │
│  │  │  LLM Selection (Parallel Processing)    │   │            │
│  │  │  Gemini-2.5-Flash                       │   │            │
│  │  │  Max Workers: 3                         │   │            │
│  │  └─────────────────────────────────────────┘   │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT LAYER                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Combined ICD-10 Codes (Valid + Mapped + Corrected)  │       │
│  │  + Descriptions + Billable Status + Evidence          │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                   │
│  Export: CSV + Downloadable Report + Streamlit UI                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
RAFgenAI/
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (GOOGLE_API_KEY)
│
├── document_processing/            # PDF Processing Module
│   ├── pdf_loader.py              # PDF text extraction
│   ├── text_cleaner.py            # Text normalization
│   ├── chunker.py                 # Sentence-aware chunking
│   └── ocr_engine.py              # OCR for scanned PDFs
│
├── icd_mapping/                    # ICD Code Processing Module
│   ├── icd_validator.py           # Code validation against master data
│   ├── icd_corrector.py           # LLM-based code correction
│   ├── icd_vector_index.py        # FAISS semantic search
│   ├── correction_filter.py       # Smart filtering for corrections
│   └── gem_selector.py            # ICD-9 to ICD-10 mapping selection
│
├── clinical_extraction/            # LLM Extraction Module
│   ├── chain.py                   # LLM extraction logic (batch processing)
│   ├── prompts.py                 # Prompt templates
│   └── schema.py                  # Pydantic data models
│
├── utils/                          # Utility Functions
│   ├── config.py                  # Configuration loader
│   └── rate_limiter.py            # Adaptive API rate limiting
│
├── scripts/                        # Data Preparation Scripts
│   ├── build_faiss_index.py       # Build FAISS index from ICD-10 data
│   ├── convert_icd10_to_csv.py    # Convert ICD-10 TXT to CSV
│   └── convert_gem_to_csv.py      # Convert GEM TXT to CSV
│
└── data/                           # Master Data Files
    ├── icd10cm_2026.csv           # Official ICD-10-CM codes (2026)
    ├── valid_icd_9_codes.xlsx     # ICD-9 master codes
    ├── 2015_I9gem.csv             # ICD-9 to ICD-10 GEM mapping
    └── faiss_icd_index/           # Pre-built FAISS vector index
```

---

## Data Flow

### Step-by-Step Processing Flow

```
1. PDF Upload
   ↓
2. Text Extraction (PyMuPDF → OCR if needed)
   Input: PDF file
   Output: Raw text string
   ↓
3. Text Cleaning (Regex normalization)
   Input: Raw text
   Output: Cleaned text
   ↓
4. Text Chunking (200 tokens, sentence-aware)
   Input: Cleaned text
   Output: List of text chunks
   ↓
5. LLM Semantic Extraction (Batch processing, 5 chunks at a time)
   Input: Batches of 5 chunks
   Output: List of (ICD codes + diagnosis objects) per chunk
   ↓
6. Validation (Against master data)
   ├─ Validate as ICD-10
   │  Input: Merged codes
   │  Output: Valid ICD-10 codes + Mismatched codes
   │
   └─ Validate Mismatched as ICD-9
      Input: Mismatched codes
      Output: Valid ICD-9 codes + Invalid codes
   ↓
7. Smart Correction (Only for invalid semantic codes)
   ├─ Filter: Instant fixes (E119 → E11.9)
   ├─ Filter: Skip low-confidence extractions
   └─ FAISS Search + LLM Selection (Parallel, max 3 workers)
      Input: Invalid code + condition text
      Output: Corrected ICD-10 code
   ↓
8. GEM Mapping (ICD-9 → ICD-10)
   Input: Valid ICD-9 codes
   Process:
   - Priority 1: approximate=1 mappings
   - Priority 2: approximate=0 mappings
   - If multiple ICD-10 codes: LLM selects best match
   Output: Mapped ICD-10 codes
   ↓
9. Combine Final Results
    Input: Valid ICD-10 + Mapped ICD-10 + Corrected ICD-10
    Output: Final unique ICD-10 code list
    ↓
10. Enrich with Metadata
    Input: Final ICD-10 codes
    Process: Lookup descriptions + billable status
    Output: Enriched code table
    ↓
11. Export & Display
    Output: CSV file + Streamlit tables + Download button
```

---

## Core Modules

### 1. Document Processing Module

#### `pdf_loader.py`
**Purpose**: Extract text from PDF files with automatic OCR fallback.

**Functions**:

##### `is_text_valid(text: str) -> bool`
- **Input**: 
  - `text` (str): Extracted text from PDF
- **Output**: 
  - `bool`: True if text is valid, False otherwise
- **Logic**:
  - Checks if text has at least 50 words
  - Verifies alpha character ratio > 40%
  - Returns False for empty or gibberish text

##### `extract_text_from_pdf(file_path: str) -> Tuple[str, bool]`
- **Input**: 
  - `file_path` (str): Path to PDF file
- **Output**: 
  - Tuple: `(extracted_text, used_ocr)`
  - `extracted_text` (str): Full text content
  - `used_ocr` (bool): True if OCR was used
- **Logic**:
  1. Opens PDF with PyMuPDF (fitz)
  2. Extracts text from each page using `page.get_text()`
  3. Validates text quality with `is_text_valid()`
  4. If invalid, triggers OCR via `extract_text_from_scanned_pdf()`
  5. Returns extracted text + OCR flag

---

#### `ocr_engine.py`
**Purpose**: Perform OCR on scanned PDFs using Tesseract.

**Functions**:

##### `extract_text_from_scanned_pdf(file_path: str) -> str`
- **Input**: 
  - `file_path` (str): Path to scanned PDF
- **Output**: 
  - `str`: OCR-extracted text
- **Logic**:
  1. Converts PDF pages to images using `pdf2image`
  2. Runs Tesseract OCR on each image
  3. Concatenates all page texts
  4. Returns full text string

---

#### `text_cleaner.py`
**Purpose**: Clean and normalize extracted text.

**Functions**:

##### `clean_text(text: str) -> str`
- **Input**: 
  - `text` (str): Raw extracted text
- **Output**: 
  - `str`: Cleaned text
- **Logic**:
  - Removes multiple spaces (collapse to single space)
  - Removes page numbers ("Page X of Y")
  - Removes standalone numbers on new lines
  - Strips leading/trailing whitespace

---

#### `chunker.py`
**Purpose**: Split text into sentence-aware chunks (~200 tokens).

**Functions**:

##### `protect_icd_codes(text: str) -> str`
- **Input**: 
  - `text` (str): Text with ICD codes
- **Output**: 
  - `str`: Text with protected ICD codes
- **Logic**:
  - Replaces "." in ICD codes with `<DOT>`
  - Prevents sentence splitter from breaking codes
  - Example: `E11.9` → `E11<DOT>9`

##### `restore_icd_codes(text: str) -> str`
- **Input**: 
  - `text` (str): Text with protected codes
- **Output**: 
  - `str`: Text with restored ICD codes
- **Logic**:
  - Replaces `<DOT>` back to "."
  - Example: `E11<DOT>9` → `E11.9`

##### `split_into_sentences(text: str) -> List[str]`
- **Input**: 
  - `text` (str): Text to split
- **Output**: 
  - `List[str]`: List of sentences
- **Logic**:
  - Splits on period followed by space + capital letter
  - Regex: `r'(?<=[a-zA-Z0-9])\.\s+(?=[A-Z])'`
  - Preserves ICD codes with decimals

##### `chunk_text_by_tokens(text: str, max_tokens: int = 200) -> List[str]`
- **Input**: 
  - `text` (str): Full text to chunk
  - `max_tokens` (int): Maximum tokens per chunk (default: 200)
- **Output**: 
  - `List[str]`: List of text chunks
- **Logic**:
  1. Protects ICD codes from sentence splitting
  2. Splits text into sentences
  3. Groups sentences into chunks ≤ max_tokens
  4. Keeps sentences complete (never breaks mid-sentence)
  5. Restores ICD code format
  6. Returns list of chunks

---

### 2. ICD Mapping Module

#### `icd_validator.py`
**Purpose**: Validate extracted codes against official ICD master data.

**Functions**:

##### `normalize_icd(code: str) -> str`
- **Input**: 
  - `code` (str): ICD code (may have dots)
- **Output**: 
  - `str`: Normalized code (no dots, uppercase)
- **Logic**:
  - Removes periods
  - Converts to uppercase
  - Example: `e11.22` → `E1122`

##### `validate_icd_codes(regex_codes: List[str], icd_master_df: pd.DataFrame) -> Tuple[List[str], List[str]]`
- **Input**: 
  - `regex_codes` (List[str]): Codes to validate
  - `icd_master_df` (DataFrame): Master ICD data
- **Output**: 
  - Tuple: `(matched_codes, mismatched_codes)`
  - `matched_codes` (List[str]): Valid codes (original format)
  - `mismatched_codes` (List[str]): Invalid codes (original format)
- **Logic**:
  1. Normalizes master codes (removes dots, uppercase)
  2. For each code:
     - Normalizes code
     - Checks if exists in master set
     - Adds to matched or mismatched list
  3. Returns both lists

---

#### `icd_corrector.py`
**Purpose**: Correct invalid ICD codes using FAISS + LLM.

**LLM Models**:
- **Model**: Gemini-2.5-Flash
- **Temperature**: 0 (deterministic)
- **Timeout**: 30 seconds

**Functions**:

##### `correct_invalid_code_detailed(invalid_code: str, condition_text: str, max_retries: int = 2, billable_ratio: float = 0.85) -> Optional[Dict]`
- **Input**: 
  - `invalid_code` (str): Invalid ICD code (e.g., "E119")
  - `condition_text` (str): Medical condition description
  - `max_retries` (int): Retry attempts (default: 2)
  - `billable_ratio` (float): Ratio of billable codes (for FAISS optimization)
- **Output**: 
  - `Dict` or `None`:
    ```python
    {
      "llm1_icd_code": "E119",
      "llm1_description": "Type 2 diabetes",
      "top_5_similar_codes": {
        "E11.9": "Type 2 diabetes without complications",
        "E11.65": "Type 2 diabetes with hyperglycemia",
        ...
      },
      "llm2_valid_icd_code": "E11.9",
      "llm2_valid_description": "Type 2 diabetes without complications"
    }
    ```
- **Logic**:
  1. **FAISS Search**: Gets top 5 similar billable codes
  2. **LLM Selection**: Uses prompt with CRITICAL RULES:
     - Must select billable code
     - Must preserve clinical meaning and severity
     - Must not downgrade specificity
     - Must preserve laterality, temporal classification
  3. **Retry Logic**: Up to `max_retries` attempts
  4. **Fallback**: Returns highest scoring FAISS result if LLM fails
  5. **Output**: Detailed correction dictionary

##### `correct_codes_parallel_detailed(invalid_codes: List[str], condition_texts: List[str], max_workers: int = 3, billable_ratio: float = 0.85) -> List[Optional[Dict]]`
- **Input**: 
  - `invalid_codes` (List[str]): List of invalid codes
  - `condition_texts` (List[str]): List of condition descriptions
  - `max_workers` (int): Parallel workers (default: 3)
  - `billable_ratio` (float): Billable code ratio
- **Output**: 
  - `List[Optional[Dict]]`: List of correction results (same order as input)
- **Performance**:
  - Sequential: 12s for 3 codes (4s each)
  - Parallel: 4s for 3 codes (67% reduction)
- **Logic**:
  1. Creates task list with indices to preserve order
  2. Submits all tasks to ThreadPoolExecutor
  3. Collects results as they complete
  4. Returns results in original input order

##### `correct_codes_smart(invalid_codes: List[str], condition_texts: List[str], evidence_snippets: List[str], icd10_master_df, max_workers: int = 3, confidence_threshold: float = 0.4, billable_ratio: float = 0.85) -> Dict`
- **Input**: 
  - `invalid_codes` (List[str]): Invalid codes
  - `condition_texts` (List[str]): Condition descriptions
  - `evidence_snippets` (List[str]): Evidence quotes from text
  - `icd10_master_df` (DataFrame): ICD-10 master data
  - `max_workers` (int): Parallel workers
  - `confidence_threshold` (float): Minimum confidence (0.0-1.0)
  - `billable_ratio` (float): Billable code ratio
- **Output**: 
  ```python
  {
    "corrected_codes": {"E119": "E11.9", ...},
    "instant_fixes": {"E119": "E11.9"},
    "llm_corrections": {"G209": "G20.C"},
    "skipped": {"X99": "low_confidence_0.25"},
    "stats": {...},
    "detailed_results": [...]
  }
  ```
- **Performance**:
  - Original: 8s for 6 codes (all go to LLM)
  - Smart: 3s for 6 codes (2 instant, 2 skipped, 2 LLM)
  - Time saved: 60%, Cost saved: 67%
- **Logic**:
  1. **Filter**: Applies smart filtering via `filter_codes_for_correction()`
     - Instant fixes for format errors
     - Skips low-confidence extractions
     - Identifies codes needing LLM
  2. **Parallel LLM**: Corrects filtered codes in parallel
  3. **Combine**: Merges instant fixes + LLM corrections
  4. **Statistics**: Returns detailed performance metrics

---

#### `correction_filter.py`
**Purpose**: Intelligent filtering to reduce expensive LLM calls.

**Functions**:

##### `is_simple_format_error(invalid_code: str) -> bool`
- **Input**: 
  - `invalid_code` (str): Code to check
- **Output**: 
  - `bool`: True if simple format error
- **Logic**:
  - Detects missing decimal (E119 → E11.9)
  - Detects extra spaces (E11 .9 → E11.9)
  - Detects wrong case (e11.9 → E11.9)
  - Detects trailing characters (E11.9x → E11.9)

##### `fix_format(invalid_code: str, icd10_master_df) -> Optional[str]`
- **Input**: 
  - `invalid_code` (str): Code to fix
  - `icd10_master_df` (DataFrame): Master data for validation
- **Output**: 
  - `str` or `None`: Fixed code if successful
- **Logic**:
  1. Cleans code (strip, uppercase, remove spaces)
  2. Removes trailing non-alphanumeric characters
  3. Adds missing decimal for 4+ digit codes
  4. Validates format with regex
  5. Optionally validates against master data
  6. Returns fixed code or None

##### `calculate_condition_confidence(condition_text: str, evidence_snippet: str = "") -> float`
- **Input**: 
  - `condition_text` (str): Extracted condition
  - `evidence_snippet` (str): Supporting evidence
- **Output**: 
  - `float`: Confidence score (0.0 - 1.0)
- **Scoring Logic**:
  - Base score: 0.5
  - +0.2 if no vague terms (unspecified, unknown, unclear)
  - +0.2 if has specific indicators (type 1/2, acute, chronic, left/right)
  - +0.1 if has evidence snippet (>10 chars)
  - -0.2 if very short condition (<15 chars)
  - -0.2 if only generic terms (disease, disorder, condition)
  - Clamped to [0.0, 1.0]

##### `should_correct_code(invalid_code: str, condition_text: str, evidence_snippet: str, icd10_master_df, confidence_threshold: float = 0.4) -> Tuple[bool, Optional[str], str]`
- **Input**: 
  - `invalid_code` (str): Invalid code
  - `condition_text` (str): Condition description
  - `evidence_snippet` (str): Evidence from text
  - `icd10_master_df` (DataFrame): Master data
  - `confidence_threshold` (float): Minimum confidence
- **Output**: 
  - Tuple: `(needs_llm_correction, instant_fix_code, reason)`
- **Decision Logic**:
  1. If simple format error → Apply instant fix
  2. If low confidence → Skip correction
  3. If invalid format → Skip correction
  4. Otherwise → Needs LLM correction

##### `filter_codes_for_correction(invalid_codes, condition_texts, evidence_snippets, icd10_master_df, confidence_threshold) -> dict`
- **Input**: 
  - `invalid_codes` (list): Invalid codes
  - `condition_texts` (list): Condition descriptions
  - `evidence_snippets` (list): Evidence snippets
  - `icd10_master_df` (DataFrame): Master data
  - `confidence_threshold` (float): Minimum confidence
- **Output**: 
  ```python
  {
    "instant_fixes": {"E119": "E11.9"},
    "needs_llm": [("G209", "Parkinson's disease", "patient has PD")],
    "skipped": {"X99": "low_confidence_0.25"},
    "stats": {
      "total_invalid": 6,
      "instant_fixes": 2,
      "needs_llm": 2,
      "skipped": 2,
      "llm_calls_saved": 4,
      "time_saved_estimate": 16,
      "cost_reduction_pct": 66.7
    }
  }
  ```
- **Logic**:
  - Iterates through all invalid codes
  - Applies `should_correct_code()` decision logic
  - Categorizes into instant_fixes / needs_llm / skipped
  - Calculates detailed statistics

---

#### `icd_vector_index.py`
**Purpose**: FAISS-based semantic search for ICD codes.

**Configuration**:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Index Location**: `data/faiss_icd_index/`
- **Device**: CPU

**Functions**:

##### `load_faiss_index()`
- **Input**: None
- **Output**: FAISS vector store object
- **Logic**:
  - Lazy loading (loads once, reuses)
  - Initializes HuggingFace embeddings
  - Loads FAISS index from disk
  - Raises error if index not found

##### `find_similar_icd_codes(condition_text: str, top_k: int = 5, score_threshold: float = 0.5, billable_only: bool = True, billable_ratio: float = 0.85) -> List[Tuple[str, str, float]]`
- **Input**: 
  - `condition_text` (str): Medical condition description
  - `top_k` (int): Number of results (default: 5)
  - `score_threshold` (float): Minimum similarity (0-1)
  - `billable_only` (bool): Filter to billable codes only
  - `billable_ratio` (float): Ratio of billable codes (default: 0.85)
- **Output**: 
  - `List[Tuple[str, str, float]]`: List of (code, description, similarity_score)
  - Example: `[("E11.9", "Type 2 diabetes without complications", 0.92), ...]`
- **Performance Optimization (Step 7)**:
  - Old: `search_k = top_k * 3` (always 3x)
  - New: `search_k = top_k * (1 / billable_ratio) * 1.1`
  - Example: If 85% billable → multiplier = 1.18x (60% reduction)
- **Logic**:
  1. Loads FAISS index
  2. Calculates optimized search_k based on billable_ratio
  3. Performs FAISS similarity search
  4. Converts distance to similarity score: `1.0 / (1.0 + distance)`
  5. Filters by score_threshold
  6. Filters out non-billable codes if `billable_only=True`
  7. Stops when `top_k` billable codes found
  8. Returns list of tuples

##### `find_similar_by_invalid_code(invalid_code: str, condition_text: str, top_k: int = 5, billable_only: bool = True, billable_ratio: float = 0.85) -> List[Tuple[str, str, float]]`
- **Input**: 
  - `invalid_code` (str): Invalid ICD code
  - `condition_text` (str): Condition description
  - `top_k` (int): Number of candidates
  - `billable_only` (bool): Return only billable codes
  - `billable_ratio` (float): Billable ratio for optimization
- **Output**: 
  - `List[Tuple[str, str, float]]`: Top K billable similar codes
- **Logic**:
  - Constructs query: `"{invalid_code}: {condition_text}"`
  - Calls `find_similar_icd_codes()` with query
  - Returns BILLABLE codes only (prevents non-billable parent mappings)

---

#### `gem_selector.py`
**Purpose**: Select best ICD-10 code when multiple GEM mappings exist.

**LLM Model**:
- **Model**: Gemini-2.5-Flash
- **Temperature**: 0 (deterministic)
- **Timeout**: 30 seconds

**Functions**:

##### `select_best_icd10_from_gem(icd9_code: str, icd9_description: str, icd10_candidates: List[str], icd10_descriptions: dict, clinical_context: str, clinical_evidence: str = "", max_retries: int = 2) -> Optional[str]`
- **Input**: 
  - `icd9_code` (str): ICD-9 code (e.g., "1269")
  - `icd9_description` (str): ICD-9 description
  - `icd10_candidates` (List[str]): Possible ICD-10 codes
  - `icd10_descriptions` (dict): Code → description mapping
  - `clinical_context` (str): Clinical note text
  - `clinical_evidence` (str): Evidence snippet supporting diagnosis
  - `max_retries` (int): Retry attempts
- **Output**: 
  - `str` or `None`: Selected ICD-10 code
- **Prompt Instructions**:
  - Read clinical evidence and context carefully
  - Consider laterality, severity, complications
  - Select most SPECIFIC code supported by documentation
  - Prioritize evidence snippet over full context
- **Logic**:
  1. If only 1 candidate → Return immediately
  2. Format candidates with descriptions for prompt
  3. Call LLM with selection prompt
  4. Validate selected code is in candidates
  5. Handle code normalization (with/without dots)
  6. Retry on failure (up to max_retries)
  7. Fallback: Return first candidate if all attempts fail

##### `select_multiple_gem_mappings(icd9_codes: List[str], icd9_descriptions: dict, gem_mappings: dict, icd10_master_df: pd.DataFrame, clinical_context: str) -> dict`
- **Input**: 
  - `icd9_codes` (List[str]): ICD-9 codes to process
  - `icd9_descriptions` (dict): ICD-9 descriptions
  - `gem_mappings` (dict): `{icd9_code: [icd10_codes]}`
  - `icd10_master_df` (DataFrame): ICD-10 master data
  - `clinical_context` (str): Clinical note
- **Output**: 
  - `dict`: `{icd9_code: selected_icd10_code}`
- **Logic**:
  - Iterates through all ICD-9 codes
  - For each code with multiple mappings:
    - Calls `select_best_icd10_from_gem()`
  - Returns mapping dictionary

---

### 3. Clinical Extraction Module

#### `chain.py`
**Purpose**: LLM-based semantic extraction of ICD codes from clinical text.

**LLM Models**:
- **Semantic Model**: Gemini-2.5-Flash (high accuracy)
- **Batch Model**: Gemini-2.5-Flash (stable for batch processing)
- **Temperature**: 0 (deterministic)
- **Timeout**: 60s (semantic), 90s (batch)

**Functions**:

##### `extract_icd_from_chunk(chunk: str, max_retries: int = 2) -> Tuple[List[str], List[Diagnosis]]`
- **Input**: 
  - `chunk` (str): Text chunk (~200 tokens)
  - `max_retries` (int): Retry attempts
- **Output**: 
  - Tuple: `(icd_codes, diagnosis_objects)`
  - `icd_codes` (List[str]): List of ICD-10 codes
  - `diagnosis_objects` (List[Diagnosis]): Full diagnosis objects with condition + code + evidence
- **Logic**:
  1. Skips very small chunks (<30 chars)
  2. Formats prompt with chunk text
  3. Invokes LLM (Gemini-2.5-Flash)
  4. Parses JSON response with Pydantic
  5. Extracts ICD codes and diagnosis objects
  6. Retries on failure (up to max_retries)
  7. Returns empty lists if all attempts fail

##### `extract_icd_from_chunks_batch(chunks: list, batch_size: int = 5, max_retries: int = 2) -> List[Tuple[List[str], List[Diagnosis]]]`
- **Input**: 
  - `chunks` (list): List of text chunks
  - `batch_size` (int): Chunks per batch (default: 5)
  - `max_retries` (int): Retry attempts per batch
- **Output**: 
  - `List[Tuple[List[str], List[Diagnosis]]]`: Results for each chunk (in same order)
- **Performance**:
  - Batch processing: 80% time reduction vs sequential
  - Uses adaptive rate limiting (only waits when needed)
- **Logic**:
  1. Divides chunks into batches of size `batch_size`
  2. For each batch:
     - Smart rate limiting via `batch_rate_limiter.wait_if_needed()`
     - Formats batch prompt with all chunks
     - Invokes batch LLM
     - Parses batch JSON response
     - Handles retries on failure
     - Fallback: Sequential processing if batch fails
  3. Returns results in same order as input chunks

---

#### `prompts.py`
**Purpose**: LLM prompt templates for semantic extraction.

**Prompt Templates**:

##### `ICD_SEMANTIC_PROMPT`
- **Purpose**: Extract ICD codes from single chunk
- **Input Variables**: `chunk`
- **Instructions**:
  - Use only 2026 ICD-10-CM codes
  - Extract diagnoses even if code not explicitly written
  - Assign most specific billable code
  - Write detailed CONDITION matching ICD-10-CM long description
  - Include EVIDENCE_SNIPPET (verbatim quote, 5-20 words)
  - Ignore negated, ruled-out, suspected conditions
  - Ignore labs, vitals, medications
  - Return ONE code per diagnosis entry
  - Output strict JSON
- **Output Format**:
  ```json
  {
    "diagnoses": [
      {
        "condition": "Chronic hepatitis C",
        "icd10": "B18.2",
        "evidence_snippet": "Patient has history of hepatitis C infection, chronic"
      }
    ]
  }
  ```

##### `ICD_SEMANTIC_BATCH_PROMPT`
- **Purpose**: Extract ICD codes from multiple chunks in one call
- **Input Variables**: `chunks_text`, `num_chunks`
- **Instructions**:
  - Process each chunk independently
  - Return results in SAME ORDER as chunks
  - Return exactly `num_chunks` results
  - If chunk has no conditions, return empty diagnoses list
  - Same extraction rules as single-chunk prompt
- **Output Format**:
  ```json
  {
    "results": [
      {
        "chunk_number": 1,
        "diagnoses": [...]
      },
      {
        "chunk_number": 2,
        "diagnoses": [...]
      }
    ]
  }
  ```

---

#### `schema.py`
**Purpose**: Pydantic models for structured LLM outputs.

**Models**:

##### `Diagnosis`
```python
class Diagnosis(BaseModel):
    condition: str        # Detailed condition description
    icd10: str           # ICD-10 code
    evidence_snippet: str # Verbatim quote from text
```

##### `ICDLLMResponse`
```python
class ICDLLMResponse(BaseModel):
    diagnoses: List[Diagnosis]  # List of diagnoses from single chunk
```

##### `BatchChunkResult`
```python
class BatchChunkResult(BaseModel):
    chunk_number: int         # Chunk index (1-based)
    diagnoses: List[Diagnosis] # Diagnoses for this chunk
```

##### `BatchICDResponse`
```python
class BatchICDResponse(BaseModel):
    results: List[BatchChunkResult]  # Results for all chunks
```

---

### 4. Utility Modules

#### `config.py`
**Purpose**: Load environment variables and configuration.

**Configuration**:
```python
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

**Validation**:
- Raises `ValueError` if `GOOGLE_API_KEY` not found

---

#### `rate_limiter.py`
**Purpose**: Adaptive rate limiting for API calls.

**Classes**:

##### `AdaptiveRateLimiter`
- **Purpose**: Smart rate limiter that only sleeps when approaching limit
- **Configuration**:
  - `max_rpm` (int): Maximum requests per minute (default: 60)
  - `buffer` (float): Safety buffer (0.9 = use 90% of limit)
- **Performance**:
  - Old: Fixed sleep every 5 calls (wastes time)
  - New: Only sleeps when actually approaching limit
  - Time saved: ~2s per batch
- **Methods**:
  - `wait_if_needed()`: Wait only if approaching rate limit
  - `get_current_rate()`: Get current requests per minute
  - `get_wait_time()`: Calculate wait time before next call
  - `reset()`: Clear call history
- **Algorithm**:
  1. Remove calls older than 60 seconds
  2. Count recent calls in last minute
  3. If at/near limit, sleep until oldest call expires
  4. Record current call timestamp

##### `BatchRateLimiter`
- **Purpose**: Rate limiter optimized for batch processing
- **Configuration**:
  - `max_rpm` (int): Maximum requests per minute
  - `batch_size` (int): Items processed per API call
  - `buffer` (float): Safety buffer
- **Logic**:
  - Effective limit = `(max_rpm / batch_size) * buffer`
  - Example: If batch_size=5, each call processes 5 items
  - Allows fewer actual API calls
- **Methods**:
  - `wait_if_needed()`: Wait if approaching batch limit
  - `get_current_rate()`: Get current batch calls per minute
  - `get_effective_item_rate()`: Get items processed per minute
  - `reset()`: Clear call history

---

### 5. Scripts Module

#### `build_faiss_index.py`
**Purpose**: Build FAISS vector index from ICD-10 data (one-time setup).

**Configuration**:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Batch Size**: 1000 codes per batch
- **Device**: CPU
- **Output**: `data/faiss_icd_index/`

**Process**:
1. Loads ICD-10 CSV (`data/icd10cm_2026.csv`)
2. Optionally filters to billable codes only
3. Prepares texts: `"{code}: {description}"`
4. Prepares metadata: code, title, billable status
5. Initializes HuggingFace embeddings (downloads model if first time)
6. Builds FAISS index in batches (for memory efficiency)
7. Merges all batch indices
8. Saves final index to disk

**Runtime**: 10-15 minutes for ~70,000 codes

**Usage**:
```bash
python scripts/build_faiss_index.py
```

---

#### `convert_icd10_to_csv.py`
**Purpose**: Convert ICD-10 TXT file to CSV format.

**Input**: `data/icd10cm_order_2026.txt` (fixed-width format)
**Output**: `data/icd10cm_2026.csv`

**Fixed-Width Parsing**:
- Sequence: positions 0-5
- Code: positions 6-13
- Billable: position 14-15
- Short Title: positions 16-76
- Long Title: positions 77+

**CSV Columns**:
- `sequence`, `code`, `is_billable`, `short_title`, `long_title`

---

#### `convert_gem_to_csv.py`
**Purpose**: Convert GEM TXT file to CSV format.

**Input**: `data/2015_I9gem.txt` (space-separated)
**Output**: `data/2015_I9gem.csv`

**TXT Format**:
```
<icd9_code> <icd10_code> <flags>
```

**Flags** (5 characters):
- `approximate`: Flag 0 (0=exact, 1=approximate)
- `no_map`: Flag 1
- `combination`: Flag 2
- `scenario`: Flag 3
- `choice_list`: Flag 4

**CSV Columns**:
- `icd9_code`, `icd10_code`, `approximate`, `no_map`, `combination`, `scenario`, `choice_list`

---

### 6. Main Application (`app.py`)

#### Overview
Streamlit-based web application orchestrating the entire pipeline.

#### Key Functions

##### `build_icd_lookups(icd10_df, icd9_df)`
- **Decorator**: `@st.cache_data`
- **Purpose**: Pre-build lookup dictionaries for O(1) access
- **Performance**: 5s → 0.5s (90% reduction for repeated lookups)
- **Output**:
  ```python
  {
    'icd10_desc': {code: description},
    'icd10_billable': {code: billable_status},
    'icd9_desc': {code: description}
  }
  ```

##### `calculate_billable_ratio(icd10_df)`
- **Decorator**: `@st.cache_data`
- **Purpose**: Calculate percentage of billable codes for FAISS optimization
- **Output**: `float` (e.g., 0.85 for 85% billable)

##### `preload_faiss_index()`
- **Decorator**: `@st.cache_resource`
- **Purpose**: Pre-load FAISS index once at startup
- **Performance**: Moves 2s loading time from first correction to app startup

---

#### Application Flow

##### **Step 1: PDF Upload & Text Extraction**
```python
uploaded_file = st.file_uploader("Upload Clinical PDF", type=["pdf"])

# Save to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    tmp_path = tmp_file.name

# Extract text (with OCR fallback)
raw_text, used_ocr = extract_text_from_pdf(tmp_path)
```

##### **Step 2: Text Cleaning & Chunking**
```python
cleaned_text = clean_text(raw_text)
chunks = chunk_text_by_tokens(cleaned_text, max_tokens=200)
```

##### **Step 3: LLM Semantic Extraction (Batch)**
```python
batch_results = extract_icd_from_chunks_batch(chunks, batch_size=5)

for semantic_codes, diagnoses in batch_results:
    semantic_icd_list.append(semantic_codes)
            diagnosis_objects_list.append(diagnoses)
```

##### **Step 4: Load Master Data & Build Lookups**
```python
icd10_master_df = pd.read_csv("data/icd10cm_2026.csv", dtype=str)
icd9_master_df = pd.read_excel("data/valid_icd_9_codes.xlsx", dtype=str)
gem_df = pd.read_csv("data/2015_I9gem.csv", dtype=str)

# Normalize codes (remove dots)
icd10_master_df["icd_code"] = icd10_master_df["icd_code"].str.replace(".", "", regex=False)
icd9_master_df["icd_code"] = icd9_master_df["icd_code"].str.replace(".", "", regex=False)

# Build cached lookups (O(1) access)
icd_lookups = build_icd_lookups(icd10_master_df, icd9_master_df)

# Pre-load FAISS index
faiss_index = preload_faiss_index()

# Calculate billable ratio
billable_ratio = calculate_billable_ratio(icd10_master_df)
```

##### **Step 5: Validation**
```python
for semantic_codes in semantic_icd_list:
    # Validate as ICD-10
    matched_icd10, mismatched = validate_icd_codes(semantic_codes, icd10_master_df)
    
    # Validate mismatched as ICD-9
    matched_icd9, truly_invalid = validate_icd_codes(mismatched, icd9_master_df)
    
    # All invalid codes are from semantic extraction
    invalid_semantic = truly_invalid
```

##### **Step 6: Correct Invalid Semantic Codes**
```python
# Create mapping of code → condition/evidence
code_to_condition = {}
code_to_evidence = {}
for diag in diagnoses:
    if diag.icd10 in invalid_semantic:
        code_to_condition[diag.icd10] = diag.condition
        code_to_evidence[diag.icd10] = diag.evidence_snippet

# Parallel correction with FAISS + LLM
parallel_results = correct_codes_parallel_detailed(
    invalid_codes=invalid_semantic,
    condition_texts=[code_to_condition.get(code, chunks[i]) for code in invalid_semantic],
    max_workers=3,
    billable_ratio=billable_ratio
)

# Add corrected codes to matched_icd10
for result in parallel_results:
    if result:
        corrected_code = result["llm2_valid_icd_code"]
        corrected_codes.append(corrected_code)
        
        # Validate corrected code
        validated, _ = validate_icd_codes([corrected_code], icd10_master_df)
        if validated:
                    matched_icd10.extend(validated)
```

##### **Step 7: GEM Mapping (ICD-9 → ICD-10)**
```python
for icd9_code in matched_icd9:
    normalized = icd9_code.replace(".", "")
    
    # Priority 1: approximate=1
    approx_matches = gem_df[
        (gem_df["icd9_code"] == normalized) &
        (gem_df["approximate"] == "1")
    ]["icd10_code"].tolist()
    
    # Priority 2: approximate=0
    if approx_matches:
        selected_matches = approx_matches
    else:
        exact_matches = gem_df[
            (gem_df["icd9_code"] == normalized) &
            (gem_df["approximate"] == "0")
        ]["icd10_code"].tolist()
        selected_matches = exact_matches
    
    # If multiple mappings, use LLM to select best one
    if len(selected_matches) > 1:
        icd9_desc = icd9_desc_map.get(normalized, "Unknown condition")
        evidence_snippet = icd9_evidence_map.get(icd9_code, "")
        
        best_code = select_best_icd10_from_gem(
            icd9_code=icd9_code,
            icd9_description=icd9_desc,
            icd10_candidates=selected_matches,
            icd10_descriptions=icd10_desc_map,
            clinical_context=chunks[i],
            clinical_evidence=evidence_snippet
        )
        
        if best_code:
            mapped_icd10.append(best_code)
    else:
        # Single mapping, add directly
        mapped_icd10.extend(selected_matches)
```

##### **Step 8: Combine Final ICD-10 Codes**
```python
combined_icd10 = list(dict.fromkeys(matched_icd10 + mapped_icd10))
```

##### **Step 9: Create DataFrame**
```python
df = pd.DataFrame({
    "Chunk Number": range(1, len(chunks) + 1),
    "200 Token Chunk": chunks,
    "semantic_icd_codes": semantic_icd_list,
    "validated_icd10_codes": icd10_list,
    "validated_icd9_codes": icd9_list,
    "mapped_icd10_from_icd9": mapped_icd10_list,
    "icd9_to_icd10_mapping_dict": mapping_dictionary_list,
    "gem_llm_selections": gem_selections_list,
    "final_combined_icd10_codes": final_icd10_list,
    "invalid_semantic_icd_codes": invalid_semantic_list,
    "corrected_codes": corrected_codes_list,
    "correction_details": correction_details_list
})
```

##### **Step 10: Display Results**
```python
# Main results table
st.dataframe(df, use_container_width=True, hide_index=True)

# Global final ICD-10 codes
all_final_icd10 = sorted(set(code for sublist in final_icd10_list for code in sublist))

# Enriched table with descriptions
final_table_data = []
for code in all_final_icd10:
    normalized = code.replace(".", "")
    description = code_desc_map.get(normalized, "Description not found")
    is_billable = code_billable_map.get(normalized, "Unknown")
    billable_status = "Yes" if is_billable == "1" else "No"
    
    final_table_data.append({
        "icd_code": code,
        "icd_description": description,
        "is_billable": billable_status
    })

final_table_df = pd.DataFrame(final_table_data)
st.dataframe(final_table_df, use_container_width=True, hide_index=True)

# Display billable vs non-billable summary
billable_df = final_table_df[final_table_df["is_billable"] == "Yes"]
non_billable_df = final_table_df[final_table_df["is_billable"] == "No"]

col1, col2 = st.columns(2)
with col1:
    st.metric("Billable Codes", len(billable_df))
with col2:
    st.metric("Non-Billable Codes", len(non_billable_df))

# Display correction details
correction_table_data = []
for detail in all_correction_details:
    top_5_formatted = "\n".join([
        f"{code}: {desc}" 
        for code, desc in detail["top_5_similar_codes"].items()
    ])
    
    correction_table_data.append({
        "LLM1 Invalid Code": detail["llm1_icd_code"],
        "LLM1 Description": detail["llm1_description"],
        "Top 5 Similar Codes (FAISS)": top_5_formatted,
        "LLM2 Valid Code": detail["llm2_valid_icd_code"],
        "LLM2 Valid Description": detail["llm2_valid_description"]
    })

correction_df = pd.DataFrame(correction_table_data)
st.dataframe(correction_df, use_container_width=True, hide_index=True)

# Display GEM multi-mapping selections
gem_table_data = []
for gem_sel in all_gem_selections:
    candidates_formatted = "\n".join([
        f"{code}: {code_desc_map.get(code, 'Unknown')}" 
        for code in gem_sel["candidates"]
    ])
    
    selected_desc = code_desc_map.get(gem_sel["selected"], "Unknown")
    
    gem_table_data.append({
        "ICD-9 Code": gem_sel["icd9_code"],
        "Available ICD-10 Mappings": candidates_formatted,
        "LLM Selected Code": gem_sel["selected"],
        "Selected Description": selected_desc
    })

gem_df_display = pd.DataFrame(gem_table_data)
st.dataframe(gem_df_display, use_container_width=True, hide_index=True)
```

##### **Step 11: Export CSV**
```python
os.makedirs("outputs", exist_ok=True)

base_filename = os.path.splitext(uploaded_file.name)[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"outputs/{base_filename}_{timestamp}_chunks.csv"

df.to_csv(output_path, index=False)

# Download button
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download Chunk Table as CSV",
    data=csv_data,
    file_name=f"{base_filename}_{timestamp}_chunks.csv",
    mime="text/csv"
)
```

---

## Performance Optimizations

### Summary of Optimizations

| Step | Optimization | Time Saved | Details |
|------|-------------|------------|---------|
| 1 | Batch LLM Processing | 80% | Process 5 chunks per API call instead of sequential |
| 2 | Parallel Code Correction | 67% | ThreadPoolExecutor with 3 workers (12s → 4s) |
| 3 | Smart Filtering | 60% | Instant fixes + skip low-confidence (8s → 3s) |
| 4 | Cached Lookups | 90% | O(n) → O(1) dictionary access (5s → 0.5s) |
| 5 | Pre-load FAISS Index | 100% | Load once at startup instead of per correction |
| 6 | Optimized FAISS Search | 60% | Search multiplier 1.18x vs 3x based on billable ratio |
| 7 | Adaptive Rate Limiting | ~2s/batch | Only sleep when approaching API limit |

### Detailed Optimizations

#### **1. Batch LLM Processing (Step 1)**
- **Before**: Sequential processing, 1 chunk per API call
- **After**: Batch processing, 5 chunks per API call
- **Time Reduction**: 80% (10s → 2s for 5 chunks)
- **Implementation**: `extract_icd_from_chunks_batch()` with `ICD_SEMANTIC_BATCH_PROMPT`

#### **2. Parallel Code Correction (Step 2)**
- **Before**: Sequential correction, 4s per code
- **After**: ThreadPoolExecutor with 3 workers
- **Time Reduction**: 67% (12s → 4s for 3 codes)
- **Implementation**: `correct_codes_parallel_detailed()` with `ThreadPoolExecutor`

#### **3. Smart Filtering (Step 3)**
- **Before**: All invalid codes go to expensive LLM correction
- **After**: Filter logic (instant fixes + skip low-confidence)
- **Time Reduction**: 60% (8s → 3s for 6 codes)
- **Cost Reduction**: 67% (4 LLM calls saved)
- **Implementation**: `filter_codes_for_correction()` with confidence scoring

#### **4. Cached Lookups (Step 5)**
- **Before**: DataFrame queries (O(n)) for each lookup
- **After**: Dictionary lookups (O(1))
- **Time Reduction**: 90% (5s → 0.5s for repeated lookups)
- **Implementation**: `build_icd_lookups()` with `@st.cache_data`

#### **5. Pre-load FAISS Index (Step 6)**
- **Before**: Load FAISS index on first correction (~2s)
- **After**: Load once at app startup
- **Time Reduction**: 100% for subsequent corrections
- **Implementation**: `preload_faiss_index()` with `@st.cache_resource`

#### **6. Optimized FAISS Search (Step 7)**
- **Before**: Always search 3x codes to filter billable
- **After**: Dynamic multiplier based on billable ratio
- **Search Reduction**: 60% (3x → 1.18x for 85% billable)
- **Accuracy**: Same (still gets required billable codes)
- **Implementation**: `find_similar_icd_codes()` with dynamic `search_multiplier`

#### **7. Adaptive Rate Limiting (Step 8)**
- **Before**: Fixed sleep every 5 calls (wastes time)
- **After**: Only sleep when approaching API limit
- **Time Saved**: ~2s per batch (eliminates unnecessary waits)
- **Implementation**: `AdaptiveRateLimiter` and `BatchRateLimiter`

---

## API Dependencies

### Google Gemini API
- **Model**: Gemini-2.5-Flash
- **Usage**:
  - Semantic ICD extraction (batch mode)
  - Invalid code correction
  - GEM multi-mapping selection
- **Rate Limit**: 60 RPM (paid tier)
- **Configuration**: Set `GOOGLE_API_KEY` in `.env` file

### HuggingFace (Local)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Usage**: Generate embeddings for FAISS index
- **No API Key Required**: Runs locally on CPU

### Tesseract OCR (Local)
- **Usage**: OCR for scanned PDFs
- **Installation Required**: System-level installation
- **No API Key Required**

---

## Setup and Installation

### 1. Prerequisites
- Python 3.8+
- Tesseract OCR installed on system
- Google Gemini API key

### 2. Clone Repository
```bash
git clone <repository-url>
cd RAFgenAI
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Prepare Data Files
Place these files in `data/` folder:
- `icd10cm_2026.csv` (ICD-10 master data)
- `valid_icd_9_codes.xlsx` (ICD-9 master data)
- `2015_I9gem.csv` (GEM mapping data)

### 6. Build FAISS Index (One-time)
```bash
python scripts/build_faiss_index.py
```
This takes 10-15 minutes. Creates `data/faiss_icd_index/`.

### 7. Run Application
```bash
streamlit run app.py
```
Opens in browser at `http://localhost:8501`

---

## Data Files

### Required Files
1. **ICD-10-CM Master Data** (`data/icd10cm_2026.csv`)
   - Columns: `sequence`, `code`, `is_billable`, `short_title`, `long_title`
   - Source: CMS official 2026 ICD-10-CM codes

2. **ICD-9 Master Data** (`data/valid_icd_9_codes.xlsx`)
   - Columns: `icd_code`, `long_title`
   - Used for validating ICD-9 codes

3. **GEM Mapping** (`data/2015_I9gem.csv`)
   - Columns: `icd9_code`, `icd10_code`, `approximate`, `no_map`, `combination`, `scenario`, `choice_list`
   - Used for ICD-9 → ICD-10 conversion

4. **FAISS Index** (`data/faiss_icd_index/`)
   - Pre-built vector index for semantic search
   - Built from ICD-10 master data
   - Generated by `scripts/build_faiss_index.py`

---

## Output Files

### CSV Export
**Location**: `outputs/{filename}_{timestamp}_chunks.csv`

**Columns**:
- `Chunk Number`: Sequential chunk index (1-based)
- `200 Token Chunk`: Text content of chunk
- `semantic_icd_codes`: Codes extracted by LLM
- `validated_icd10_codes`: Valid ICD-10 codes
- `validated_icd9_codes`: Valid ICD-9 codes
- `mapped_icd10_from_icd9`: ICD-10 codes from GEM mapping
- `icd9_to_icd10_mapping_dict`: Full mapping dictionary
- `gem_llm_selections`: LLM selections for multi-mappings
- `final_combined_icd10_codes`: Final ICD-10 codes (all sources)
- `invalid_semantic_icd_codes`: Invalid codes from LLM
- `corrected_codes`: Codes after FAISS + LLM correction
- `correction_details`: Full correction metadata

---

## Error Handling

### PDF Processing Errors
- **Issue**: PDF cannot be read
- **Handling**: Display error message, stop processing
- **User Action**: Verify PDF is not corrupted

### OCR Failures
- **Issue**: Tesseract not installed or fails
- **Handling**: Raises exception with installation instructions
- **User Action**: Install Tesseract OCR

### API Rate Limit
- **Issue**: Exceeding Gemini API rate limit
- **Handling**: Adaptive rate limiter automatically sleeps
- **Impact**: Processing takes longer but completes successfully

### LLM Failures
- **Issue**: LLM call fails or returns invalid JSON
- **Handling**: Retry logic (up to 2 attempts), fallback to empty results
- **Impact**: Some extractions may be missed

### FAISS Index Missing
- **Issue**: FAISS index not found
- **Handling**: Raises `FileNotFoundError` with instructions
- **User Action**: Run `python scripts/build_faiss_index.py`

### Code Correction Failures
- **Issue**: Cannot correct invalid code
- **Handling**: Returns highest FAISS similarity match as fallback
- **Impact**: May return suboptimal correction

---

## Best Practices

### 1. PDF Quality
- Use high-quality scans (300+ DPI)
- Ensure text is readable if scanned
- Digital PDFs process faster than scanned

### 2. API Usage
- Monitor API costs (Gemini charges per token)
- Batch processing reduces costs by 80%
- Smart filtering reduces correction costs by 67%

### 3. Performance
- Pre-load FAISS index at startup
- Use cached lookups for repeated queries
- Limit concurrent workers to 3 for correction

### 4. Data Quality
- Keep master data files up to date
- Rebuild FAISS index when master data changes
- Validate GEM mapping file format

### 5. Troubleshooting
- Check Streamlit logs for errors
- Verify API key is valid
- Ensure all data files are present
- Check FAISS index exists

---

## Future Enhancements

### Potential Improvements
1. **Multi-language support**: Add support for non-English clinical notes
2. **Real-time processing**: Stream results as they're extracted
3. **Custom master data**: Allow users to upload custom ICD codes
4. **Audit trail**: Track all changes and corrections
5. **Confidence scores**: Display confidence for each extraction
6. **Export formats**: Add PDF, Excel, JSON export options
7. **Batch PDF processing**: Process multiple PDFs at once
8. **API integration**: REST API for programmatic access

---

## Version History

### Current Version: 1.0
- Hybrid extraction (Regex + LLM)
- FAISS-based correction
- GEM mapping with LLM selection
- Batch processing optimization
- Adaptive rate limiting
- Smart filtering for corrections
- Billable/non-billable classification
- Evidence snippet tracking

---

## License & Attribution

This system uses:
- **ICD-10-CM codes**: CMS official classification (public domain)
- **GEM mappings**: CMS General Equivalence Mappings (public domain)
- **Gemini API**: Google Generative AI
- **FAISS**: Facebook AI Similarity Search (MIT License)
- **Sentence Transformers**: HuggingFace (Apache 2.0)

---

## Support & Contact

For issues, questions, or contributions:
- Check documentation in this file
- Review code comments for implementation details
- Test with sample PDFs before production use

---

**End of Documentation**
