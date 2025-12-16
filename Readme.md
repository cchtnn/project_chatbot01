```markdown
# Jericho Enterprise RAG Chatbot

Multi‑domain RAG chatbot (HR, payroll, student policies, transcripts) built with FastAPI, ChromaDB, and open‑source LLM tooling.  
Tested with 36+ Diné College documents (PDF/CSV/DOCX/JSON), ~3,800 chunks indexed.

---

## 1. Prerequisites (Windows / VDI)

### 1.1 System

- Windows 10/11
- Python 3.10 (recommended)
- Git
- VS Code (optional but recommended)

### 1.2 Tesseract OCR

Used for OCR on scanned PDFs.

1. Download installer (Windows):
   - https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default path:
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 1.3 Poppler (for pdf2image)

Used for rasterizing PDFs before OCR when needed.

1. Download Windows build:
   - https://github.com/oschwartz10612/poppler-windows/releases
2. Extract somewhere, e.g.:
   - `C:\tools\poppler-24.02.0\`
3. Add to PATH (PowerShell, per‑session):
```

$env:PATH = "C:\tools\poppler-24.02.0\Library\bin;$env:PATH"

```

---

## 2. Clone and create virtualenv

Open PowerShell in VS Code (or terminal) and run:

```

# 1) Clone

git clone <YOUR-REPO-URL> jericho-backend
cd jericho-backend

# 2) Create venv

python -m venv venv_p310

# 3) Activate venv

.\venv_p310\Scripts\Activate.ps1

# 4) Upgrade pip and install deps

pip install --upgrade pip
pip install -r requirements.txt

```

Ensure `requirements.txt` includes (among others):

```

fastapi
uvicorn[standard]
chromadb
sentence-transformers
pypdf
pdfplumber
python-docx
openpyxl
python-pptx
markdown
pytesseract
pillow
pdf2image
pandas
numpy<2.0.0
groq
langchain
langchain-community
langchain-ollama

```

---

## 3. Environment configuration

Project uses `.env` and/or OS environment variables for secrets and paths.

### 3.1 .env file (recommended)

Create a file `.env` in the project root:

```

# LLM provider

LLM_PROVIDER=groq

# Groq LLM

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Optional: Ollama fallback (if running locally)

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Tesseract path (Windows)

TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

```

### 3.2 Alternatively, set env vars in PowerShell

```

$env:GROQ_API_KEY = "your_groq_key_here"
$env:LLM_PROVIDER = "groq"
$env:GROQ_MODEL = "llama-3.1-8b-instant"

$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:PATH = "C:\tools\poppler-24.02.0\Library\bin;$env:PATH"

```

---

## 4. Tesseract configuration in code

In `services/document_parser.py` the hardcoded path:

```

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

```

has been replaced with env‑based config:

```

import os
import pytesseract

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

```

Make sure `TESSERACT_CMD` is set before running the app.

---

## 5. Initial document ingest

This builds the ChromaDB index from documents (PDF/CSV/DOCX/JSON) under `data/documents/`.

From the repo root (where `ingest_all.py` lives):

```

.\venv_p310\Scripts\Activate.ps1
cd backend # if your code is under backend/, adjust if needed

python ingest_all.py

```

You should see logs like:

- `ChromaDB ready: 0 chunks`
- `Ingested ...`
- `ChromaDB ready: 3830 chunks`

If you change documents, re‑run `ingest_all.py` to update the index.

---

## 6. Run the FastAPI app

From the same venv:

```

.\venv_p310\Scripts\Activate.ps1
cd backend

uvicorn app:app --reload --port 8000

```

Then open in browser:

- Chat UI: `http://localhost:8000/chat`
- Admin UI: `http://localhost:8000/admin`
- OpenAPI docs: `http://localhost:8000/docs`

---

## 7. Quick backend sanity test

There is a small script (e.g. `temp.py`) that checks retrieval and answers:

```

from core import init_logger
init_logger()
from services.rag_pipeline import RAGPipeline

rag = RAGPipeline()

queries = [
"What is the check date for Pay period 3?",
"How many payroll periods are in calendar year 2026?",
"What are the employee health benefits?",
"What is the student code of conduct?",
]

print("GENERIC RAG TEST ACROSS ALL FORMATS\n")

for q in queries:
result = rag.query(q, top_k=10)
print("Q:", q)
print(" Agent:", result.get("agent"))
print(" Answer:", result.get("answer", "")[:300])
print()

```

Run:

```

python temp.py

```

You should see:

- `Enterprise RAG v2.0 ready - Multi-Modal`
- Reasonable answers with `agent` values like `text_rag` or `table_agent`.

---

## 8. Using the chat UI

1. Start the server (`uvicorn app:app --reload --port 8000`).
2. Open `http://localhost:8000/chat`.
3. Create a new session (handled automatically by UI).
4. Ask questions such as:
   - “What is the check date for Pay period 3?”
   - “How many payroll periods are in calendar year 2026?”
   - “What employee health benefits are provided?”
   - “Summarize the student code of conduct.”

Answers are produced by the `RAGPipeline` using hybrid retrieval + multi‑modal agents.

---

## 9. Uploading new documents

You can upload additional documents via the UI:

1. Go to `http://localhost:8000/chat`.
2. Use the upload control to send PDF/CSV/DOCX/JSON files.
3. Backend saves to `data/documents/` and ingests through `RAGPipeline.ingest_documents`.

Or, place files manually under `data/documents/` and run:

```

python ingest_all.py

```

---

## 10. Running tests (optional)

If you add tests under `tests/`, you can run:

```

.\venv_p310\Scripts\Activate.ps1
pytest -q

```

---

## 11. Common issues

- **Tesseract not found**:
  Ensure Tesseract is installed and `TESSERACT_CMD` is set to the correct `tesseract.exe` path.

- **Poppler missing** (`pdf2image` errors):
  Ensure Poppler `bin` folder is on PATH before running `ingest_all.py`.

- **Groq 401/403**:
  Check `GROQ_API_KEY` and `GROQ_MODEL` values in `.env`.

- **No answers / hallucinations**:
  Re‑run `python ingest_all.py` after updating documents. Confirm `ChromaDB ready: N chunks` is non‑zero.

---

## 12. Project structure (high level)

```

backend/
app.py # FastAPI app
api/
routes.py # /query, /upload, sessions
services/
rag_pipeline.py # Enterprise RAG v2.0 (multi-modal)
document_parser.py # Multi-format parsing + OCR
text_processor.py # Chunking
core/
config.py # Settings / env
retrieval.py # Hybrid retriever (BM25 + vector)
embeddings.py # Sentence-transformers wrapper
db/
chromadb_manager.py # Vector store
data/
documents/ # Source docs
vectorstore/ # ChromaDB files
logs/ # Logs

```

This is enough to clone, set up, ingest, and run the chatbot on any new Windows VDI with Python 3.10.
```
