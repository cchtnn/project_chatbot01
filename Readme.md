# Jericho Enterprise RAG Chatbot

## 1. Setting up on a new machine / VDI

> Repo: https://github.com/yesitsrg/project_chatbot01.git  
> Root folder after clone: `jericho/` (backend + frontend inside)

### 1.1 Prerequisites (once per VDI)

- Windows 10/11
- Python 3.10 (recommended, 64-bit)
- Node.js 18+ (for React frontend)
- Git installed
- Internet access (for `pip` / `npm` and Groq API, if used)

#### Install Tesseract OCR

1. Download Windows installer (UB Mannheim build):  
   https://github.com/UB-Mannheim/tesseract/wiki  
2. Install to:  
   `C:\Program Files\Tesseract-OCR\tesseract.exe` (default)  
3. Optional quick check (PowerShell):

   ```
   & "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   ```

#### Install Poppler (for pdf2image)

1. Download Windows build:  
   https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to e.g.:  
   `C:\poppler-24.02.0\Library\bin\pdftoppm.exe`
3. Add Poppler bin to PATH for current session:

   ```
   $env:PATH += ";C:\poppler-24.02.0\Library\bin"
   ```

### 1.2 Clone repo

```
# From any folder where you keep projects
git clone https://github.com/yesitsrg/project_chatbot01.git jericho
cd jericho
```

Directory (expected):

```
jericho/
  backend/
  frontend/
  ingestall.py
  Readme.md  (this file)
  data/      (created later if not present)
```

### 1.3 Backend setup (FastAPI + RAG)

```
cd backend

# Create & activate venv
python -m venv venvp310
.\venvp310\Scripts\Activate.ps1

# Install backend dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Set environment variables (for this PowerShell session):

```
# Tesseract path
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"

# Poppler path (already added above, repeat if new session)
$env:PATH += ";C:\poppler-24.02.0\Library\bin"

# Groq API key (optional if using Groq)
$env:GROQ_API_KEY = "your-groq-key"
$env:LLMPROVIDER = "groq"    # or "ollama" for local models
```

Alternatively, create `.env` in `backend/`:

```
LLMPROVIDER=groq
GROQ_API_KEY=your-groq-key
GROQ_MODEL=llama-3.1-8b-instant
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 1.4 Documents + ingestion

On your original machine you already ingested 36 Diné College files (HR, payroll, transcripts, etc.) into Chroma. On a new machine, copy those source documents into the repo and re-ingest.

```
cd backend

# Ensure documents folder exists
New-Item -ItemType Directory -Force -Path data\documents | Out-Null

# Copy your domain docs (from old VDI / shared drive) into:
# jericho\backend\data\documents\
# e.g.
#   data\documents\hrpolicies\*.pdf
#   data\documents\payroll\*.csv
#   data\documents\transcripts\*.csv
#   data\documents\borplanner\*.pdf

# Build vector index (ChromaDB)
python ingestall.py
# Expect logs like:
# "INGEST STATS processed 36, failed 0"
# "ChromaDB ready XXXX chunks"
```

### 1.5 Start backend API

```
cd backend
.\venvp310\Scripts\Activate.ps1

uvicorn app:app --reload --port 8000
```

Check:

```
Invoke-RestMethod -Uri http://localhost:8000/api/v1/health
# Should return JSON with status, chunk counts, providers
```

### 1.6 Frontend setup (React)

```
cd ..\frontend

npm install
npm start
```

- React dev server: http://localhost:3000
- It calls backend at http://localhost:8000 (CORS enabled in `backend/app.py`)

---

## 2. Project structure (high level)

```
jericho/
├── backend/
│   ├── app.py               # FastAPI app, mounts API & CORS
│   ├── config.py            # Settings (env, paths, LLM provider)
│   ├── core/
│   │   ├── logger.py        # Structured logging (file + console)
│   │   ├── constants.py     # Enums, dirs, chunk sizes, modes
│   │   ├── embeddings.py    # SentenceTransformer wrapper
│   │   └── retrieval.py     # Hybrid BM25 + vector retrieval
│   ├── models/
│   │   ├── schemas.py       # Pydantic API models
│   │   └── document.py      # SQLAlchemy + Pydantic doc metadata
│   ├── services/
│   │   ├── documentparser.py    # Multi-format parser + OCR chain
│   │   ├── textprocessor.py     # Chunking / cleaning
│   │   ├── dataviews.py         # Payroll / transcript table loaders
│   │   ├── dfagent.py           # Pandas DataFrame agent wrapper
│   │   ├── tools/
│   │   │   ├── transcripttool.py  # Transcript-specific agent
│   │   │   ├── payrolltool.py     # Payroll calendar agent
│   │   │   └── ...                # BOR / planner tools, etc.
│   │   ├── orchestrator.py     # Agentic routing + answer synthesis
│   │   └── ragpipeline.py      # RAG pipeline (ingest + query)
│   ├── api/
│   │   ├── health.py           # /api/v1/health
│   │   └── routes.py           # /api/v1/query, /upload, sessions*
│   ├── db/
│   │   └── chromadbmanager.py  # ChromaDB wrapper
│   ├── data/
│   │   ├── documents/          # Source docs (copied by you)
│   │   └── vectorstore/        # Chroma persistence
│   └── ingestall.py            # Bulk ingest script
└── frontend/
    ├── src/                    # React components
    ├── package.json
    └── ...                     # Standard React app layout
```

Endpoints used by frontend (aligned with original app):

- `POST /api/v1/query` – main chat endpoint (form/json; includes `query`, `sessionid`, `private`)
- `POST /api/v1/upload` – upload docs into KB
- `POST /api/v1/newsession`, `GET /api/v1/usersessions`, etc. – session management (where implemented)
- `GET /api/v1/health` – health check

---

## 3. Git remote / repo notes

Your current repo is:

```
origin  https://github.com/yesitsrg/project_chatbot01.git (fetch)
origin  https://github.com/yesitsrg/project_chatbot01.git (push)
```

If this is correct (and matches GitHub), no change is needed.

If you ever need to fix it:

```
cd jericho

# See current remotes
git remote -v

# Reset origin
git remote remove origin
git remote add origin https://github.com/yesitsrg/project_chatbot01.git

# First commit & push (only once)
git add .
git commit -m "Initial commit Jericho enterprise RAG chatbot"
git branch -M main
git push -u origin main
```

On a new VDI, you just:

```
git clone https://github.com/yesitsrg/project_chatbot01.git jericho
# then follow Section 1
```

---

## 4. Enterprise RAG techniques used (with examples & impact)

This project is intentionally “enterprise-grade” and incorporates several advanced RAG design patterns. Here is what to study and where it shows up in the code.

### 4.1 Multi-format, fault-tolerant parsing (DocumentParser)

**What:**  
A “bulletproof” parser that handles many formats and recovers from bad PDFs:

- PDF: `pymupdf` → `pdfplumber` (text+tables) → `pdf2image + Tesseract` OCR fallback
- DOCX: `python-docx` (paragraphs, tables)
- CSV/XLSX: `pandas.read_csv/read_excel` → markdown tables
- JSON: `json` → pretty-printed text
- Images: `pytesseract` then `easyocr` fallback
- PPTX: text extraction with fallback to plain text

**Where:** `backend/services/documentparser.py`  
**Why:** Some Diné College PDFs are scanned or have broken text layers. This chain:

- Greatly reduces “file failed to ingest” cases (you saw ~97% success vs many failures earlier).
- Extracts tables into structured text so payroll and transcript data are usable.

**Impact (observed earlier):**

- Previously failed PDFs (catalog, handbook) now parse, adding ~2000+ extra chunks.
- Ingestion stats improved from ~77–80% success to ~97–100% for your test corpus.

### 4.2 Semantic chunking with metadata (TextProcessor)

**What:**

- Splits content into chunks of ~1000 characters with 200-character overlap.
- Respects simple semantic boundaries (paragraph / heading breaks).
- Attaches detailed metadata per chunk (document id, filename, page number, domain, etc.).

**Where:** `backend/services/textprocessor.py`  

**Why:**

- Smaller chunks → more precise retrieval (less noise).
- Overlap → enough context for the LLM to answer without missing the surrounding text.
- Metadata → enables source citations and domain filtering (e.g., only HR, only payroll).

**Impact:**

- Better answer grounding (easier to show “this line came from HR policy PDF page 7”).
- Enabled future features like domain-scoped search (HR vs transcripts vs BOR).

### 4.3 Hybrid retrieval (BM25 + vector + rerank)

**What:**

- Uses semantic vector search (ChromaDB + SentenceTransformers) AND keyword-based BM25.
- Merges results using Reciprocal Rank Fusion (RRF).
- Optionally reranks using a cross-encoder (e.g., ms-marco based model).

**Where:** `backend/core/retrieval.py`, `backend/db/chromadbmanager.py`, `backend/services/ragpipeline.py`

**Why:**

- Pure embeddings miss exact numeric / keyword queries (e.g., “pay period 3”, dates).
- Pure BM25 misses semantic / paraphrased queries.
- Hybrid ensures both precise matching and semantic understanding.

**Impact:**

- Higher recall for tricky queries (payroll dates, specific terms).
- More relevant top-5 chunks for the LLM to answer from, reducing hallucinations.

### 4.4 Domain-specific agents (“agentic RAG”)

**What:**

Instead of a single generic RAG, the system routes queries to specialized tools:

- TranscriptTool – operates on merged transcript CSV (GPA, courses, counts).
- PayrollTool – operates on payroll calendar CSV (pay periods, check dates, counts).
- BOR/PlannerTool – operates on BOR schedule / planner.
- Generic RAG – for policy, handbook, catalog-type questions.

A planner/orchestrator decides which tool to call based on query content.

**Where:**

- `backend/services/tools/transcripttool.py`
- `backend/services/tools/payrolltool.py`
- `backend/services/orchestrator.py`
- `backend/services/dataviews.py` (loads dataframes)
- `backend/services/dfagent.py` (pandas DataFrame agent for free-form table questions)

**Examples:**

- “What is the check date for Pay period 3?” → PayrollTool:
  - Interpret “Pay period 3” → row in payroll CSV → return check date for period 3.
- “Give me the GPA details of Trista Barrett.” → TranscriptTool:
  - Filter transcript dataframe by student name → return term + cumulative GPA.

**Why:**

- Generic RAG on CSV text is unreliable for structured questions (dates, counts, numeric queries).
- Agents let you use pandas & explicit logic for tables, while still using LLM for explanation.

**Impact (from your benchmarks):**

- Payroll questions that previously failed (“Unable to process…” or wrong dates) can now be answered correctly.  
- Transcript questions avoid NaN / nonsense GPA hallucinations and provide structured answers.

### 4.5 Dataframe agent (LangChain + pandas)

**What:**

- Wraps a pandas DataFrame in a LangChain “pandas dataframe agent” so LLM can generate code-like operations over the table to answer questions.
- Used as a fallback in TranscriptTool / PayrollTool when no simple rule applies.

**Where:** `backend/services/dfagent.py`  

**Why:**

- You cannot hand-code every possible analytic query (“top N students”, “average GPA by term”) for transcripts / payroll.
- The dataframe agent can handle flexible slice/filter/aggregation queries using LLM-powered pandas code, constrained to your data.

**Impact:**

- Better support for analytic queries (counts, averages, grouped summaries) without writing many custom functions.
- Still keeps data grounded because the agent only uses the loaded dataframe.

*(Note: depends on `langchain-experimental` and the associated agent toolkit.)*

### 4.6 Multi-provider LLM configuration

**What:**

- Config-driven LLM selection:
  - `LLMPROVIDER=groq` → use Groq-hosted models (e.g., Llama 3.1).
  - `LLMPROVIDER=ollama` → use local Ollama models (e.g., `llama3latest`).
- Configured via `config.py` + `.env`.

**Where:** `backend/config.py`, `backend/services/ragpipeline.py`, `backend/services/llmfactory.py` (if present)

**Why:**

- You can run fully local (Ollama) or cloud-inference (Groq) without changing code.
- Easier migration across environments: dev VDI vs production server.

**Impact:**

- Flexibility for cost/performance/security trade-offs.
- You already tested this by switching between Ollama and Groq during development.

---

## 5. What was wrong in the old project, and how it is solved now

### 5.1 Old issues

1. **Parsing reliability**
   - Many PDFs (scanned, complex layout) failed to parse or produced garbage text.
   - Important docs (catalog, handbook, HR policies) partially or not at all ingested.
   - No proper OCR fallback chain.

2. **Weak retrieval**
   - Pure vector search (or naive retrieval) missed:
     - Numeric queries (pay period numbers, dates).
     - Exact-match queries on table columns.
   - Retrieval only on text, not on structured CSV/Excel.

3. **No domain separation**
   - One generic RAG path for everything.
   - Transcript, payroll, BOR planner, and policy questions all went through the same generic answer prompt.
   - This caused:
     - Hallucinated GPA lists (NaN/0.00).
     - Wrong or missing payroll dates.
     - Generic answers where precise answers were required.

4. **Auth and benchmarking difficulty**
   - Old HTML/Jinja-based app had login, cookies, and session DB.
   - Automated benchmarking via scripts was hard because:
     - `query` required JWT cookies + real `sessionid`.
   - Hard to systematically compare old vs new behavior.

5. **Git / project structure confusion**
   - Nested git repo in `backend/` made the top-level repo unmanageable.
   - Remote URL mismatch led to push errors.
   - README / setup instructions were outdated relative to the new architecture.

### 5.2 New solutions

1. **Robust parser + OCR chain**
   - `DocumentParser` now:
     - Uses multiple fallbacks (pymupdf → pdfplumber → pdf2image + Tesseract).
     - Handles CSV, DOCX, PPTX, images, etc.
   - Result: ingest success >95% on your corpus; previously failing PDFs now load correctly.

2. **Hybrid retrieval + ChromaDB**
   - ChromaDB for embeddings + BM25 for keywords.
   - RRF / reranking improves top-k relevance.
   - Result: better recall for both “concept” and “exact-field” queries.

3. **Agentic, domain-specific tools**
   - TranscriptTool, PayrollTool, BOR/PlannerTool, GenericRAG.
   - Tools query structured CSV/Excel dataframes and/or Chroma context.
   - Orchestrator routes queries by type.
   - Result:
     - Payroll questions handled by dedicated logic (date normalization, pay period mapping).
     - Transcript questions use structured transcript data.
     - Policy questions still use generic RAG.

4. **Evaluation and debugging**
   - You created benchmark questions (Book3.csv etc.) and scripts (`benchmark.py`) to:
     - Call new backend.
     - Compare answers with old system and/or expected answers.
   - Debug scripts (`debugquery.py`, `debugpayrollbranch.py`, etc.) to isolate failures.
   - Result: iterative tuning rather than guesswork.

5. **Clean Git & repo organization**
   - Removed inner `.git` from `backend` so `jericho/` is a single repo.
   - Correct `origin` remote: `https://github.com/yesitsrg/project_chatbot01.git`.
   - This README is up to date for:
     - New clone on any VDI.
     - Backend + frontend setup.
     - Where to look in code for each RAG technique.

---

## 6. How to study and extend the enterprise RAG design

If you want to deepen your understanding or extend this system:

- **Study**:
  - `services/documentparser.py` – multi-format parsing, OCR fallbacks.
  - `core/retrieval.py` + `db/chromadbmanager.py` – hybrid retrieval + Chroma integration.
  - `services/tools/transcripttool.py` and `services/tools/payrolltool.py` – domain agents.
  - `services/dfagent.py` – pandas dataframe agent for table QA.
  - `services/orchestrator.py` + `services/ragpipeline.py` – overall RAG orchestration.

- **Extend**:
  - Add new domain agents (e.g., “FinanceTool”, “HRBenefitsTool”) that operate on new CSV/PDF sources.
  - Add evaluation scripts to track metrics (answer accuracy, groundedness) before/after changes.
  - Enhance UI to expose domain filters or debug views (show retrieved chunks, table previews).

This structure is now stable enough that you can clone it into any Windows VDI, follow the setup steps, and expect the same behavior as on your current machine.
