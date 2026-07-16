# AI Ingestion Engine

A multimodal **RAG (Retrieval-Augmented Generation)** pipeline that turns messy source files — PDFs and videos — into clean, searchable knowledge, then answers questions about them with **cited sources** (page numbers for PDFs, timestamps for videos).

For videos it goes further: it **reads the slides** (vision model), **transcribes the speech** (Whisper), can generate **Minutes of Meeting** and email them, and answers questions through a conversational **AI Agent** (in n8n).

The focus of this project is the **ingestion engine**, not the chat: the pipeline that extracts, cleans, chunks, embeds, and stores content correctly. Good retrieval starts with good ingestion.

---

## What it does

Upload a PDF or a video → the engine extracts the content → splits it into meaningful chunks with metadata → embeds each chunk → stores it in a vector database. Then you can search it semantically, ask a cited natural-language question, chat with an agent, or (for videos) generate emailed meeting minutes.

| Input | What the engine does | Stored with |
|---|---|---|
| Normal PDF | Extract selectable text page by page | page number |
| Scanned PDF | Render pages to images, run OCR | page number + `extraction_method` |
| Video (frames) | Sample a frame every N seconds; a **vision model** reads on-screen text **and describes** the frame (OCR fallback); skip near-duplicates | timestamp + frame number |
| Video (audio) | Extract audio and **transcribe the speech** (Whisper) into timestamped segments | timestamp |
| Question | Embed it, retrieve top chunks, answer from them only | source citations |
| Meeting video | Pull the transcript back and generate **Minutes of Meeting** (map-reduce for long ones) | emailed via n8n |

---

## Architecture: one engine, many extractors

PDF and video are handled by **different extractors** but flow into a **single shared downstream pipeline** — chunk → embed → store → search → answer. The paths converge at a shared `_ingest_texts` seam in `ingestion_service.py`, so the core engine is written once.

```
        PDF file                         Video file
           |                                 |
   +-------v--------+          +-------------v--------------+
   |  PDF parser    |          |  Video parser              |
   |  (PyMuPDF)     |          |   - frames (OpenCV)        |
   |  + OCR fallback|          |     -> vision describe     |
   |  (Tesseract)   |          |        (OCR fallback)      |
   |                |          |   - audio -> Whisper       |
   +-------+--------+          +-------------+--------------+
           |                                 |
           +----------------+----------------+
                            |   _ingest_texts  (shared seam)
                            v
                  chunk -> embed -> store
                            |
                            v
                   +--------------+
                   |    Qdrant    |  vectors + metadata payloads
                   +------+-------+
                          |
          +-----------+---+----+-------------+
          v           v        v             v
       /search      /ask    /minutes    n8n AI Agent
   (semantic top-k) (RAG)  (meeting     (chat -> /agent/search
                            minutes)      -> cited answers)
```

**One-sentence version:** the system turns PDFs and videos into searchable chunks, stores them in Qdrant, and lets an LLM (or a chat agent) answer questions with page or timestamp sources — and can summarize meeting videos into emailed minutes.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| API | **FastAPI** (uvicorn) | Async, auto-generated Swagger docs at `/docs` |
| Vector DB | **Qdrant** (Docker) | Semantic search + metadata filtering; `content_chunks`, 768-dim, cosine |
| Embeddings | **Ollama** local — `nomic-embed-text` | Runs locally, 768-dim, no API cost |
| LLM (answers, minutes) | **Ollama Cloud** — `gpt-oss:20b-cloud` | Capable model without local GPU |
| Transcription | **OpenAI Whisper** (`whisper-1`) | Accurate speech-to-text with timestamps |
| Vision (frame description) | **OpenAI** `gpt-4o-mini` | Reads on-screen text + describes visuals |
| OCR (fallback) | **Tesseract** (pytesseract) | Scanned PDFs, images in PDFs, video frame text |
| PDF parsing | **PyMuPDF** | Fast text extraction + page rasterization |
| Video frames / audio | **OpenCV** + **ffmpeg** (imageio-ffmpeg) | Frame sampling; audio extraction |
| Automation & agent | **n8n** (Docker) | Upload/ingest/minutes workflows + chat AI Agent |
| Runtime | **Python 3.12** in `.venv` | |

> **Note on the vision model:** the instructor suggested Qwen vision on Ollama Cloud, but that model required a paid subscription and a local Qwen vision model was too large for this machine — so frame description uses OpenAI `gpt-4o-mini` (reusing the Whisper key). Same capability; the vision client is an isolated module, so swapping models is a one-file change.

---

## Quickstart

**Prerequisites:** Python 3.12, Docker, [Ollama](https://ollama.com), Tesseract OCR, and an **OpenAI API key** (for transcription + vision).

```bash
# 1. Clone
git clone https://github.com/jalalalarab/ai-ingestion-engine.git
cd ai-ingestion-engine

# 2. Start Qdrant and n8n (Docker)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n:latest
# daily use afterwards: docker start qdrant n8n

# 3. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt

# 4. Config — copy the example and fill in your OPENAI_API_KEY
copy .env.example .env              # Windows  (cp on macOS/Linux)

# 5. Pull the embedding model; sign in for cloud LLM
ollama pull nomic-embed-text
ollama signin

# 6. Run the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

All settings are documented in **`.env.example`** — including `OPENAI_API_KEY` (required for transcription/vision), `VIDEO_SAMPLE_SECONDS`, `TRANSCRIBE_VIDEO`, `DESCRIBE_FRAMES`, `VISION_MODEL`, and `MOM_BATCH_CHARS`. To run without OpenAI, set `TRANSCRIBE_VIDEO=false` and `DESCRIBE_FRAMES=false` (video falls back to Tesseract OCR only).

---

## API endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/ingest/pdf` | Upload + ingest a PDF (OCR fallback for scans) |
| `POST` | `/ingest/video` | Upload + ingest a video (vision frame description + audio transcript) |
| `POST` | `/search` | Semantic top-k retrieval |
| `POST` | `/ask` | Full RAG: retrieve -> LLM -> answer + sources |
| `POST` | `/agent/search` | Agent-friendly retrieval: cited chunk text for the n8n AI Agent |
| `POST` | `/minutes/{file_id}` | Generate Minutes of Meeting from a video's transcript |

**Example — ingest a video:**
```json
// POST /ingest/video
{ "file_id": "06a3...", "source_type": "video",
  "frames_ingested": 8, "transcript_segments": 12, "chunks_created": 22 }
```

**Example — Minutes of Meeting:**
```json
// POST /minutes/06a3...
{ "file_name": "Meeting.mp4", "minutes": "1. Overview...\n5. Action Items...",
  "batches_used": 1, "method": "single-pass" }
```

---

## How it works (the parts that make it not a toy)

**Multimodal video ingestion.** A video is understood three ways, all timestamped: (1) frames sampled every `VIDEO_SAMPLE_SECONDS` (default 5) go to a **vision model** that reads on-screen text *and* describes the frame — charts, layout, scene — falling back to Tesseract OCR if vision is off or a call fails; (2) the **audio track** is extracted with ffmpeg and transcribed by **Whisper** into timestamped segments; (3) near-duplicate frames are dropped. So you can ask about what was *shown* and what was *said*, with a timestamp back.

**Minutes of Meeting with map-reduce.** `/minutes/{file_id}` pulls a video's full transcript from Qdrant and produces structured minutes (Overview, Attendees, Key Points, Decisions, Action Items). Because a long meeting's transcript can exceed the LLM's **context window**, it uses **map-reduce**: if the transcript fits, one call; if not, summarize batches independently (*map*), then combine the partial summaries (*reduce*). Works for a 2-minute stand-up or a 2-hour meeting.

**Conversational AI Agent (n8n).** A chat-triggered agent uses a custom tool that calls `/agent/search`, retrieving cited chunk text from Qdrant. It answers only from retrieved passages and cites file + page/timestamp. The retrieved chunk text is visible in n8n's execution log, so every answer is auditable. Windowed memory keeps recent turns for follow-ups without overflowing the context window.

**Semantic chunking.** The default chunker embeds every sentence, finds where the *meaning* shifts (cosine distance between neighbours), and splits at the biggest topic jumps — adapting per file rather than using a fixed threshold. Switchable via `CHUNKING_STRATEGY`. Sentence embeddings run concurrently to stay fast on CPU-only Ollama.

**Content-hash deduplication.** The `file_id` is a SHA-256 hash of the file content (folded into a UUID). Qdrant point IDs are built from it, so re-ingesting the same file **overwrites in place** instead of duplicating. Edit the file -> hash changes -> treated as new.

**Two-layer anti-hallucination guard.** On `/ask`: (1) a **confidence guard** — if the best chunk's score is below `0.45`, the LLM is never called; (2) a **prompt guard** — the system prompt restricts the model to the provided context. Every answer carries its sources.

---

## n8n workflows

Exported workflow JSON lives in `n8n_workflows/`:

- **Ingest webhook** (`ingest_webhook.json`) — POST a file to a webhook; routes to `/ingest/pdf` or `/ingest/video` by file type.
- **AI Agent** (`ai_agent_workflow.json`) — chat trigger -> AI Agent (Ollama chat model + memory) -> HTTP tool calling `/agent/search`. Answers from Qdrant with citations.
- **Upload + Minutes email** (`upload_ingest_minutes_workflow.json`) — a form to upload a PDF or video -> ingest -> if it's a video, generate minutes and email them via Gmail.

n8n runs in Docker, so it reaches the host API via `host.docker.internal:8000` (not `localhost`).

---

## Project structure

```
app/
  main.py                  # FastAPI entry point
  config.py                # all config from .env, no hardcoded values
  api/                     # routes_health, routes_ingest, routes_search,
                           #   routes_ask, routes_agent, routes_minutes
  services/                # ingestion_service, search_service, answer_service,
                           #   minutes_service
  parsers/                 # pdf_parser, ocr_parser, video_parser,
                           #   audio_extractor, vision_client
  transcription/           # transcription_client (Whisper)
  chunking/                # simple_chunker + semantic_chunker
  embeddings/              # embedding_client
  vector_store/            # qdrant_store  (the only module that talks to Qdrant)
  llm/                     # llm_client
n8n_workflows/             # exported workflow JSON
storage/uploads/           # uploaded files (gitignored)
screenshots/               # proof-of-work evidence
tests/                     # test artifacts + generators
```

---

## Limitations & next steps

Honest about what an MVP this is:

- **No background job queue** — large videos are processed in the request; long files (many frames x vision calls, plus transcription) can be slow or hit HTTP timeouts. Next: async job queue with status polling.
- **Cloud dependency & cost** — transcription and vision call OpenAI (small per-video cost); the answer LLM uses Ollama Cloud, which can occasionally return transient 5xx errors. Vision falls back to OCR on failure; other calls would benefit from retries.
- **Minutes are generated on demand**, not stored back in Qdrant — so the agent answers from the raw transcript, not the polished minutes. Storing minutes as chunks is an easy future addition.
- **Regex sentence splitting** in the semantic chunker can mis-handle abbreviations; a proper NLP tokenizer would be more robust.
- **No auth on the API** — fine for local/demo; would add API keys before deployment.

---

## What I learned building this

- Designing a pipeline around a **shared seam** so very different inputs (PDF, video frames, audio) reuse the same embed/store/search core.
- Treating the **context window** as a first-class constraint — RAG retrieval for questions, **map-reduce** for whole-transcript summarization.
- The difference between **retrieval and memory** in an agent, and why documents belong in the vector store, not chat memory.
- Practical **RAG guardrails** — confidence thresholds and prompt constraints — to stop confident-but-wrong answers.
- **Adapting under constraints** — swapping a gated/heavy vision model for an available one without touching the rest of the pipeline (isolated client module).
- Wiring **Docker networking** correctly (n8n reaching the host via `host.docker.internal`).
