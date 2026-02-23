# RAF ICD Extraction System

## 🏥 Overview
An AI-powered system that automatically extracts ICD-10 codes from clinical documents using advanced NLP, semantic search, and LLM-based reasoning. The system provides complete transparency with evidence snippets and detailed reasoning for every extracted code.

---

## 🏗️ Architecture

### Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│                    Port: 4000                               │
│  - File upload UI                                           │
│  - Service health monitoring                                │
│  - Results display with evidence & reasoning                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ HTTP/REST
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend (Spring Boot)                          │
│                    Port: 9090                               │
│  - File handling & validation                               │
│  - Request routing                                          │
│  - Response aggregation                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ HTTP/REST (multipart/form-data)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Python FastAPI Service                         │
│                    Port: 8500                               │
│  - PDF/Document text extraction                             │
│  - LLM-based semantic extraction                            │
│  - ICD validation & mapping                                 │
│  - FAISS semantic search                                    │
│  - Complete provenance tracking                             │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- React 18
- Axios for HTTP
- CSS3 with animations

**Backend:**
- Spring Boot 3.4.1
- Java 24
- Maven
- Jackson for JSON

**AI/ML Service:**
- Python 3.10+
- FastAPI
- Google Gemini 2.5 Pro & Flash
- LangChain
- FAISS (Facebook AI Similarity Search)
- Sentence Transformers (all-MiniLM-L6-v2)

---

## 🔍 How ICD Codes Are Found

### The 7-Step Pipeline

```
Document → [1] → [2] → [3] → [4] → [5] → [6] → [7] → Final ICD-10 Codes
```

#### **Step 1: Document Processing & Text Extraction**
- **Input:** PDF, TXT, DOC, or DOCX file
- **Process:** 
  - Extract text using PyPDF2 with OCR fallback (Tesseract)
  - Clean text (remove noise, normalize whitespace)
  - Chunk text into 200-token segments for LLM processing
- **Output:** List of cleaned text chunks

#### **Step 2: LLM Semantic Extraction (Batch Processing)**
- **Input:** Text chunks
- **Process:**
  - Process 5 chunks per LLM call (batch optimization)
  - Gemini 2.5 Pro extracts:
    - Clinical conditions
    - ICD-10 codes (initial extraction)
    - Evidence snippets (direct quotes)
  - Uses structured Pydantic schema for validation
- **Prompt:** `ICD_SEMANTIC_BATCH_PROMPT`
- **Output:** List of `Diagnosis` objects (condition, icd10, evidence_snippet)

#### **Step 3: ICD-10 Validation**
- **Input:** Extracted ICD codes
- **Process:**
  - Validate against ICD-10-CM 2026 codebook (93,000+ codes)
  - Check format and existence
  - Normalize codes (remove dots: E11.9 → E119)
- **Output:** 
  - ✅ Valid ICD-10 codes
  - ❌ Invalid/unmatched codes → move to Step 4

#### **Step 4: ICD-9 Fallback Validation**
- **Input:** Invalid codes from Step 3
- **Process:**
  - Check if codes are valid ICD-9 codes
  - Validate against ICD-9 codebook
- **Output:**
  - ✅ Valid ICD-9 codes → move to Step 5 (GEM mapping)
  - ❌ Truly invalid codes → move to Step 6 (FAISS correction)

#### **Step 5: GEM Mapping (ICD-9 → ICD-10)**
- **Input:** Valid ICD-9 codes
- **Process:**
  - Use GEM (General Equivalence Mappings) - official CMS mapping
  - Priority: `approximate = 1` mappings
  - **If 1 ICD-10 mapping:** Direct conversion
  - **If multiple ICD-10 mappings:** LLM selection
    - Gemini 2.5 Flash evaluates clinical context
    - Considers laterality, severity, specificity
    - Returns selected code + detailed reasoning
- **Prompt:** `GEM_SELECTION_PROMPT`
- **Output:** ICD-10 codes + reasoning for multi-candidate scenarios

#### **Step 6: FAISS Semantic Correction**
- **Input:** Truly invalid codes (not ICD-9 or ICD-10)
- **Process:**
  1. **FAISS Vector Search:**
     - Query: Invalid code + condition description
     - Search: 93,000 ICD-10 embeddings (Sentence-BERT)
     - Return: Top 5 most similar valid codes
  2. **LLM Selection:**
     - Gemini 2.5 Flash evaluates top 5 candidates
     - Considers clinical meaning, specificity, severity
     - Returns corrected code + detailed reasoning
- **Prompt:** `FAISS_CORRECTION_PROMPT`
- **Output:** Corrected ICD-10 codes + reasoning

#### **Step 7: Global Reconciliation**
- **Input:** All extracted codes from Steps 3, 5, 6
- **Process:**
  - Analyze all diagnoses together
  - Remove duplicates
  - Merge related conditions
  - Select most specific codes
  - Verify consistency across document
- **Prompt:** `ICD_GLOBAL_RECONCILIATION_PROMPT`
- **Output:** Final deduplicated list of ICD-10 codes with complete provenance

---

## 💬 Prompts Explained

### 1. **ICD_SEMANTIC_BATCH_PROMPT**
**Purpose:** Initial extraction of conditions and ICD codes from clinical text

**Used In:** Step 2 (LLM Semantic Extraction)

**What It Does:**
- Extracts all medical conditions from text chunks
- Assigns ICD-10 codes to each condition
- Captures evidence snippets (direct quotes)
- Processes 5 chunks in parallel for efficiency

**Example Output:**
```json
{
  "results": [
    {
      "chunk_number": 1,
      "diagnoses": [
        {
          "condition": "Type 2 diabetes mellitus",
          "icd10": "E119",
          "evidence_snippet": "Patient has elevated HbA1c of 8.2%"
        }
      ]
    }
  ]
}
```

### 2. **GEM_SELECTION_PROMPT**
**Purpose:** Choose the correct ICD-10 code when ICD-9 has multiple mappings

**Used In:** Step 5 (GEM Mapping)

**What It Does:**
- Receives ICD-9 code + description
- Evaluates multiple ICD-10 candidates from GEM
- Considers clinical context and evidence
- Selects most specific/accurate ICD-10 code
- Provides detailed reasoning

**Example Scenario:**
- ICD-9: `25000` (Diabetes without complications)
- GEM Returns: `E109` (Type 1) or `E119` (Type 2)
- LLM analyzes clinical note
- Selects `E119` based on "adult-onset, non-insulin dependent" evidence

### 3. **FAISS_CORRECTION_PROMPT**
**Purpose:** Correct invalid ICD codes using semantic similarity

**Used In:** Step 6 (FAISS Semantic Correction)

**What It Does:**
- Receives invalid code + condition description
- Gets top 5 similar codes from FAISS
- Evaluates semantic similarity to clinical meaning
- Ensures corrected code preserves:
  - Clinical accuracy
  - Specificity level
  - Severity classification
  - Anatomical location
- Returns corrected code + reasoning

**Example Scenario:**
- Invalid Code: `E1199` (doesn't exist)
- Condition: "Type 2 diabetes with kidney complications"
- FAISS Top 5: `E119`, `E1121`, `E1122`, `E1129`, `E139`
- LLM selects `E1122` (with diabetic CKD)
- Reasoning explains why kidney complications require specific code

### 4. **ICD_GLOBAL_RECONCILIATION_PROMPT**
**Purpose:** Final pass to deduplicate and refine all extracted codes

**Used In:** Step 7 (Global Reconciliation)

**What It Does:**
- Reviews all diagnoses from all chunks together
- Merges duplicate or overlapping conditions
- Selects most specific codes (e.g., E1122 over E119)
- Removes redundant parent codes
- Ensures consistency across document

---

## 🔄 ICD-9 to ICD-10 Mapping (GEM)

### What is GEM?

**GEM (General Equivalence Mappings)** is the official mapping system published by CMS (Centers for Medicare & Medicaid Services) for converting between ICD-9 and ICD-10 codes.

### Why ICD-9?

Even though ICD-10 is the current standard:
- Legacy systems still use ICD-9
- Old medical records contain ICD-9 codes
- Some LLMs trained on older data may output ICD-9
- Historical data requires conversion

### GEM Mapping Process

```
ICD-9 Code → GEM Database → ICD-10 Candidate(s)
```

#### Mapping Types:

1. **1:1 Mapping (Simple)**
   ```
   ICD-9: 1234 → ICD-10: A123
   ```
   - Direct conversion
   - No ambiguity
   - Automatic mapping

2. **1:Many Mapping (Complex)**
   ```
   ICD-9: 25000 → ICD-10: [E109, E119, E139, ...]
   ```
   - Multiple possible ICD-10 codes
   - Requires clinical context to choose
   - **LLM evaluates and selects best match**

#### GEM Fields Used:

- `icd9_code`: Source code
- `icd10_code`: Target code(s)
- `approximate`: 
  - `1` = High confidence mapping (preferred)
  - `0` = Lower confidence mapping (fallback)

### Our Implementation:

1. Check GEM for `approximate = 1` mappings first
2. If none, check `approximate = 0` mappings
3. If single candidate: Direct conversion
4. If multiple candidates: **LLM Selection**
   - Send all candidates to Gemini
   - Include clinical context + evidence
   - LLM returns selected code + reasoning
   - Example reasoning: *"Selected E119 over E109 because clinical note states 'non-insulin dependent' and 'adult-onset', indicating Type 2 diabetes"*

---

## 🔎 FAISS: Semantic Search for Code Correction

### What is FAISS?

**FAISS (Facebook AI Similarity Search)** is a library for efficient similarity search in high-dimensional vector spaces. We use it to find semantically similar ICD codes.

### How We Use FAISS

#### 1. **Index Creation (One-Time Setup)**

```python
# Load all 93,000+ ICD-10 codes
icd10_codes = load_icd10_codebook()

# Create embeddings using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
for code in icd10_codes:
    text = f"{code['code']}: {code['description']}"
    embedding = model.encode(text)  # 384-dimensional vector
    
# Build FAISS index
index = faiss.IndexFlatL2(384)  # L2 distance metric
index.add(all_embeddings)

# Save index
faiss.write_index(index, "faiss_index.bin")
```

#### 2. **Search Process (Runtime)**

```
Invalid Code + Condition → Embedding → FAISS Search → Top 5 Codes → LLM Selection
```

**Example:**

```python
# Invalid code extraction
invalid_code = "E1199"  # Doesn't exist
condition = "Type 2 diabetes with kidney complications"

# Create query embedding
query_text = f"{invalid_code}: {condition}"
query_embedding = model.encode(query_text)

# Search FAISS index
distances, indices = faiss_index.search(query_embedding, k=5)

# Results (ordered by similarity)
1. E1122 - Type 2 diabetes with diabetic CKD (distance: 0.23)
2. E1121 - Type 2 diabetes with nephropathy (distance: 0.31)
3. E119  - Type 2 diabetes without complications (distance: 0.45)
4. E1129 - Type 2 diabetes with kidney complications (distance: 0.48)
5. E139  - Other diabetes with kidney complications (distance: 0.52)
```

#### 3. **LLM Final Selection**

- FAISS returns top 5 **semantically similar** codes
- LLM evaluates clinical accuracy:
  - Selects `E1122` because evidence mentions "CKD stage 3"
  - Provides reasoning: *"Selected E1122 over E1121 because the clinical note specifically mentions chronic kidney disease (CKD), not just general nephropathy"*

### Why FAISS?

1. **Speed:** Search 93,000 codes in milliseconds
2. **Accuracy:** Semantic similarity finds clinically relevant codes
3. **Robustness:** Handles typos, format errors, and near-misses
4. **No Training Required:** Uses pre-trained embeddings

### Where FAISS is Used:

- **Step 6:** Correcting truly invalid codes
- **Smart Filtering:** Finding similar codes for validation
- **Alternative Codes:** Suggesting related codes (future feature)

---

## 📊 Complete Pipeline Example

### Input Document:
```
Patient presents with elevated fasting glucose of 156 mg/dL 
and HbA1c of 8.2%. History of type 2 diabetes mellitus.
Also has stage 3 CKD related to diabetes.
```

### Pipeline Execution:

**Step 1:** Text extracted and chunked

**Step 2:** LLM extracts:
- Diagnosis 1: "Type 2 diabetes" → `E119`
- Diagnosis 2: "Diabetic CKD stage 3" → `E1122`

**Step 3:** Validate codes
- `E119`: ✅ Valid
- `E1122`: ✅ Valid

**Step 4-6:** No ICD-9 or invalid codes, skipped

**Step 7:** Global reconciliation
- Keep both codes (distinct conditions)
- Final: `E119`, `E1122`

### Final Output:

```json
{
  "icd_codes": [
    {
      "icd_code": "E119",
      "icd_description": "Type 2 diabetes mellitus without complications",
      "is_billable": "Yes",
      "evidence_snippet": "History of type 2 diabetes mellitus",
      "llm_reasoning": "Directly extracted and validated against ICD-10-CM codebook...",
      "chart_date": "2026-02-23"
    },
    {
      "icd_code": "E1122",
      "icd_description": "Type 2 diabetes with diabetic CKD",
      "is_billable": "Yes",
      "evidence_snippet": "stage 3 CKD related to diabetes",
      "llm_reasoning": "Directly extracted and validated against ICD-10-CM codebook...",
      "chart_date": "2026-02-23"
    }
  ]
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Java 24
- Node.js 16+
- Maven 3.9+
- Google API Key (Gemini)

### Installation

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Automatic_Medical_Coder
   ```

2. **Setup Python Environment**
   ```bash
   python -m venv rafenv
   rafenv\Scripts\activate  # Windows
   cd ai_icd_extraction
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   ```bash
   # Create .env file in ai_icd_extraction/
   echo GOOGLE_API_KEY=your_api_key_here > .env
   ```

4. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   ```

5. **Build Backend**
   ```bash
   cd backend
   mvn clean compile
   ```

### Running the Application

**Option 1: Automatic (Recommended)**
```bash
START_ALL_SERVICES.bat
```

**Option 2: Manual**

Terminal 1 (Python):
```bash
cd ai_icd_extraction
..\rafenv\Scripts\activate
python fastapi_service.py
```

Terminal 2 (Spring Boot):
```bash
cd backend
mvn spring-boot:run
```

Terminal 3 (React):
```bash
cd frontend
npm start
```

### Access Application

- Frontend: http://localhost:4000
- Backend API: http://localhost:9090
- Python API: http://localhost:8500
- API Docs: http://localhost:8500/docs

---

## 📂 Project Structure

```
Automatic_Medical_Coder/
├── ai_icd_extraction/              # Python FastAPI service
│   ├── scripts/
│   │   ├── clinical_extraction/    # LLM extraction & prompts
│   │   │   ├── prompts.py         # All LLM prompts
│   │   │   ├── chain.py           # LLM chains
│   │   │   └── schema.py          # Pydantic models
│   │   ├── document_processing/    # PDF/text processing
│   │   ├── icd_mapping/           # Validation & correction
│   │   │   ├── gem_selector.py    # ICD-9 → ICD-10 mapping
│   │   │   ├── icd_corrector.py   # FAISS correction
│   │   │   └── icd_vector_index.py # FAISS index
│   │   └── utils/                 # Utilities
│   ├── data/                       # ICD codebooks & GEM
│   ├── fastapi_service.py         # Main FastAPI app
│   └── response_builder.py        # Provenance tracking
├── backend/                        # Spring Boot backend
│   └── src/main/java/com/raf/icd/
│       ├── controller/            # REST controllers
│       ├── service/               # Business logic
│       └── dto/                   # Data transfer objects
├── frontend/                       # React frontend
│   └── src/
│       ├── App.js                 # Main component
│       └── App.css                # Styles
├── START_ALL_SERVICES.bat         # Start script
├── STOP_ALL_SERVICES.bat          # Stop script
└── README.md                       # This file
```

---

## 🔒 Security

- `.env` files contain API keys - **NEVER commit to Git**
- `.gitignore` configured to exclude sensitive files
- All API endpoints use CORS protection
- File uploads validated for type and size

---

## 📈 Performance

- **Batch Processing:** 5 chunks per LLM call (80% speedup)
- **Parallel Correction:** Multi-threaded FAISS corrections
- **Smart Rate Limiting:** Adaptive rate limiter for API calls
- **Caching:** FAISS index pre-loaded at startup

---

## 🎯 Key Features

- ✅ Complete provenance tracking
- ✅ Evidence snippets for every code
- ✅ Detailed LLM reasoning
- ✅ Service health monitoring
- ✅ Real-time status updates
- ✅ Automatic ICD-9 to ICD-10 conversion
- ✅ Semantic code correction with FAISS
- ✅ Beautiful, responsive UI

---

## 📝 License

[Your License Here]

---

## 🤝 Contributing

[Contributing guidelines]

---

## 📧 Contact

[Your contact information]
