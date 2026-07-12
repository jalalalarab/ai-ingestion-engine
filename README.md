# AI Ingestion Engine

A multimodal **RAG (Retrieval-Augmented Generation)** pipeline that turns messy source files — PDFs and videos — into clean, searchable knowledge, then answers questions about them with **cited sources** (page numbers for PDFs, timestamps for videos).

The focus of this project is the **ingestion engine**, not the chat: the pipeline that extracts, cleans, chunks, embeds, and stores content correctly. Good retrieval starts with good ingestion.

---

## What it does

Upload a PDF or a video → the engine extracts the text (including OCR for scanned pages and video frames) → splits it into meaningful chunks with metadata → embeds each chunk → stores it in a vector database. Then you can search it semantically, or ask a natural-language question and get an answer grounded in the stored content, with the exact page or timestamp it came from.

| Input | What the engine does | Stored with |
|---|---|---|
| Normal PDF | Extract selectable text page by page | page number |
| Scanned PDF | Render pages to images, run OCR | page number + `extraction_method` |
| Video | Sample frames, OCR the on-screen text, skip near-duplicate frames | timestamp + frame number |
| Question | Embed it, retrieve top chunks, answer from them only | source citations |

---

## Architecture: one engine, two extractors

PDF and video are handled by **different extractors** but flow into a **single shared downstream pipeline** — chunk → embed → store → search → answer. The two paths converge at a shared `_ingest_texts` seam in `ingestion_service.py`, so the core engine is written once.

```
        PDF file                    Video file
           │                            │
   ┌───────▼────────┐          ┌────────▼─────────┐
   │  PDF parser    │          │  Video parser    │
   │  (PyMuPDF)     │          │  (OpenCV frames) │
   │  + OCR fallback│          │  + Tesseract OCR │
   │  (Tesseract)   │          │                  │
   └───────┬────────┘          └────────┬─────────┘
           │                            │
           └────────────┬───────────────┘
                        │   _ingest_texts  (shared seam)
                        ▼
              chunk → embed → store
                        │
                        ▼
                 ┌──────────────┐
                 │    Qdrant    │  vectors + metadata payloads
                 └──────┬───────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
          /search               /ask
      (semantic top-k)   (retrieve → LLM → answer + sources)
```

**One-sentence version:** the system turns PDFs and videos into searchable chunks, stores them in Qdrant, and lets an LLM answer questions with page or timestamp sources.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| API | **FastAPI** (uvicorn) | Async, auto-generated Swagger docs at `/docs` |
| Vector DB | **Qdrant** (Docker) | Semantic search + metadata filtering; collection `content_chunks`, 768-dim, cosine |
| Embeddings | **Ollama** local — `nomic-embed-text` | Runs locally, 768-dim, no API cost |
| LLM (answers) | **Ollama Cloud** — `gpt-oss:20b-cloud` | Capable model without local GPU |
| OCR | **Tesseract** (pytesseract) | Scanned PDFs, images in PDFs, video frame text |
| PDF parsing | **PyMuPDF** | Fast text extraction + page rasterization for OCR |
| Video frames | **OpenCV** | Frame sampling from video |
| Automation | **n8n** (Docker) | Webhook-triggered ingestion workflows |
| Runtime | **Python 3.12** in `.venv` | |

---

## Quickstart

**Prerequisites:** Python 3.12, Docker, [Ollama](https://ollama.com), and Tesseract OCR installed on your machine.

```bash
# 1. Clone
git clone https://github.com/jalalalarab/ai-ingestion-engine.git
cd ai-ingestion-engine

# 2. Start Qdrant and n8n as Docker containers
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
docker run -d --name n8n -p 5678:5678 \
  -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n:latest

# 3. Python environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 4. Config — copy the example and adjust if needed (defaults work out of the box)
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux

# 5. Pull the embedding model into local Ollama
ollama pull nomic-embed-text
# The answer model (gpt-oss:20b-cloud) runs via Ollama Cloud.
# Sign in once so the local daemon can reach cloud models:
ollama signin

# 6. Run the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `QDRANT_COLLECTION` | `content_chunks` | Collection name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama daemon |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDING_DIM` | `768` | Vector size (pinned; must match the collection) |
| `CHUNK_SIZE_TOKENS` | `700` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `100` | Overlap between chunks |
| `MAX_PDF_MB` | `50` | Upload size cap |
| `LLM_MODEL` | `gpt-oss:20b-cloud` | Answer model |
| `LLM_TIMEOUT_SECONDS` | `120` | LLM call timeout |

---

## API endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/ingest/pdf` | Upload + ingest a PDF (with OCR fallback) |
| `POST` | `/ingest/video` | Upload + ingest a video (frame OCR) |
| `POST` | `/search` | Semantic top-k retrieval (test retrieval before answers) |
| `POST` | `/ask` | Full RAG: retrieve → LLM → answer + sources |

**Example — ingest a PDF:**
```bash
curl.exe -F "file=@storage/uploads/mydoc.pdf" http://localhost:8000/ingest/pdf
```
```json
{ "file_id": "ab10ecf2-...", "file_name": "mydoc.pdf", "source_type": "pdf",
  "pages_processed": 10, "chunks_created": 10, "ocr_pages_count": 10 }
```

**Example — ask a question:**
```json
// POST /ask
{ "question": "What did the document say about revenue growth?" }
// -> { "answer": "...", "sources": [ { "file_name": "...", "page_number": 5 } ] }
```

---

## How it works (the parts that make it not a toy)

**OCR fallback.** If a PDF page yields almost no selectable text (a scanned page), the engine renders that page to an image and runs Tesseract OCR instead — so scanned documents are still searchable. Each chunk records its `extraction_method` for debugging.

**Content-hash deduplication.** The `file_id` is derived from a SHA-256 hash of the file's content (folded into a UUID), not a random value. Because Qdrant point IDs are built from the `file_id`, re-ingesting the same file lands on the same point IDs and **overwrites in place** instead of creating duplicates. Same file in → same chunks, no bloat. Edit the file → the hash changes → it's correctly treated as new.

**Two-layer anti-hallucination guard.** On `/ask`:
1. **Confidence guard (code):** if the best retrieved chunk's similarity score is below a `0.45` threshold, the engine does **not** call the LLM at all — it returns a "not found in the provided documents" style answer. Weak context never becomes a confident wrong answer.
2. **Prompt guard (instruction):** the system prompt tells the model to answer **only** from the provided context and to say it doesn't know otherwise.

Every answer carries its **sources** (page number for PDFs, timestamp for videos), so answers are auditable.

---

## Build phases

The project was built incrementally, one working slice at a time:

- **Phase 0** — Scaffold, Docker services, `/health`, Qdrant collection at 768 dims
- **Phase 1** — PDF text ingestion → chunk → embed → store (`POST /ingest/pdf`)
- **Phase 2** — Semantic search (`POST /search`)
- **Phase 3** — RAG answers with anti-hallucination guard + cited sources (`POST /ask`)
- **Phase 4** — OCR fallback for scanned PDFs (Tesseract)
- **Phase 5** — Video ingestion: frame sampling + OCR + near-duplicate skip (`POST /ingest/video`)
- **Phase 6** — n8n automation: Webhook → HTTP Request → FastAPI → Respond to Webhook
- **Phase 7** — Logging, content-hash dedup, docs

---

## Project structure

```
app/
  main.py                  # FastAPI entry point
  config.py                # all config from .env, no hardcoded values
  logging_config.py        # one-call logging setup
  api/                     # routes_health, routes_ingest, routes_search, routes_ask
  services/                # ingestion_service, search_service, answer_service
  parsers/                 # pdf_parser, ocr_parser, video_parser
  chunking/                # simple_chunker
  embeddings/              # embedding_client
  vector_store/            # qdrant_store  (the only module that talks to Qdrant)
  llm/                     # llm_client
storage/uploads/           # uploaded files (gitignored)
screenshots/               # proof-of-work evidence
tests/                     # test artifacts + generators
```

---

## Limitations & next steps

Honest about what an MVP this is:

- **No background job queue** — very large videos/PDFs are processed in the request; long files could hit HTTP timeouts. Next: async job queue with status polling.
- **Simple chunking** — currently paragraph/token-based with overlap. Next: semantic chunking that keeps headings with their sections.
- **Single embedding model, English-focused OCR** — Tesseract language packs and multilingual embeddings would extend it.
- **No auth on the API** — fine for local/demo; would add API keys before any real deployment.
- **Vision captions for frames** are stubbed as a future step — currently video relies on frame OCR only.

---

## What I learned building this

- Designing a pipeline around a **shared seam** so two very different inputs (PDF, video) reuse the same embed/store/search core.
- Why **idempotency** matters in a data pipeline, and how a content hash gives it for free.
- Practical **RAG guardrails** — retrieval confidence thresholds and prompt constraints — to stop confident-but-wrong answers.
- Wiring **Docker networking** correctly (n8n in a container reaching FastAPI on the host via `host.docker.internal`, not `localhost`).
