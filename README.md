# AI Ingestion Engine

A multimodal **RAG (Retrieval-Augmented Generation)** pipeline that turns messy source files — PDFs and videos — into clean, searchable knowledge, then answers questions about them with **cited sources** (page numbers for PDFs, timestamps for videos).

For videos it goes further: it **reads the slides** (vision model), **transcribes the speech** (Whisper), can generate **Minutes of Meeting** and email them, and answers questions through a conversational **AI Agent** (in n8n).

It also builds a **knowledge graph** from what it ingests: an LLM extracts entities and relationships, they are loaded into **Neo4j**, and a **GraphRAG** retriever answers relationship questions by combining vector search with graph traversal. Densely-connected clusters are auto-detected (**GDS Louvain**) and summarized into high-level "community summaries."

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
| Any ingested file | Extract **(subject, predicate, object) triples** and build a **Neo4j knowledge graph** | nodes + relationships |
| Relationship question | Retrieve similar chunks **and** traverse the graph, merge both (**GraphRAG**) | passages + relationship chains |
| Whole knowledge base | Detect communities (**GDS Louvain**) and summarize each cluster | "long documents" per theme |

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
   (semantic top-k) (RAG)  (meeting     (chat: 4 tools ->
                            minutes)      search / graph / themes / minutes)

  --- knowledge graph layer (built from the same chunks) ---

   transcript chunks --> LLM triple extraction (/extract)
                              |
                              v
                      +---------------+
                      |    Neo4j      |  entities (nodes) + relationships (edges)
                      +------+--------+
                             |
              +--------------+----------------+
              v                               v
        GraphRAG (/graphrag/ask)        Communities (/graph/communities)
   vector search + graph traversal      GDS Louvain -> LLM cluster summaries
        merged -> grounded answer          ("long documents" per theme)
```

**One-sentence version:** the system turns PDFs and videos into searchable chunks, stores them in Qdrant, and lets an LLM (or a chat agent) answer questions with page or timestamp sources — and can summarize meeting videos into emailed minutes.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| API | **FastAPI** (uvicorn) | Async, auto-generated Swagger docs at `/docs` |
| Vector DB | **Qdrant** (Docker) | Semantic search + metadata filtering; `content_chunks`, 768-dim, cosine |
| Graph DB | **Neo4j 5** + **GDS** (Docker) | Knowledge graph storage + **Louvain** community detection |
| Extraction & GraphRAG LLM | **OpenAI** `gpt-4o-mini` | Triple extraction, community summaries, GraphRAG answers (strong at structured JSON + bilingual) |
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
# Neo4j with the Graph Data Science plugin (for community detection).
# The plugin value is passed via an env-file to avoid shell-quoting issues:
#   echo NEO4J_PLUGINS=["graph-data-science"] > neo4j.env
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/testpassword123 --env-file neo4j.env neo4j:5
# daily use afterwards: docker start qdrant n8n neo4j

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
| `POST` | `/minutes/by-name?name=` | Minutes by (partial) video name — one call, name in, minutes out |
| `GET`  | `/documents?name=` | List ingested files (optional name filter) — resolve a name to its `file_id` |
| `POST` | `/extract/{file_id}` | Extract (subject, predicate, object) triples from a file's transcript |
| `POST` | `/graph/build/{file_id}` | Load a file's triples into Neo4j (nodes + relationships) |
| `GET`  | `/graph/stats` | Node and relationship counts |
| `POST` | `/graphrag/ask` | GraphRAG answer: vector search **+** graph traversal (JSON body) |
| `POST` | `/graphrag/ask-simple?question=` | Same as `/graphrag/ask`, question as a query param (used by the n8n agent) |
| `POST` | `/graph/communities?min_size=` | Detect communities (Louvain) and summarize each into a "long document" |
| `POST` | `/graph/resolve-entities?apply=` | Propose (dry-run) or apply merges of duplicate entity nodes |

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

**Conversational AI Agent (n8n).** A chat-triggered agent picks between **four tools**: `search_knowledge_base` (semantic passage lookup), `ask_knowledge_graph` (GraphRAG for relationship questions), `get_theme_summaries` (community summaries for broad questions), and `get_meeting_minutes` (generate + email minutes by video name). It answers only from what the tools return and cites file + page/timestamp. Retrieved content is visible in n8n's execution log, so every answer is auditable. Windowed memory keeps recent turns without overflowing the context window.

**Entity & relationship extraction.** `/extract/{file_id}` sends the transcript to `gpt-4o-mini` in **JSON mode** (guaranteed parseable output) and pulls **triples** — `(subject, predicate, object)`. The prompt enforces a controlled predicate vocabulary and Title-Case, full-name entities so the graph doesn't fragment into synonyms; a code-side canonical map is a deterministic backstop. Batched like minutes, then merged and de-duplicated across batches.

**Knowledge graph (Neo4j).** `/graph/build/{file_id}` loads the triples into Neo4j: each entity is a `MERGE`d node (so the same entity across files collapses to one), each triple a relationship carrying its predicate and source `file_id` for provenance. Re-running clears that file's edges first, so a rebuild is clean rather than additive.

**GraphRAG retrieval — vector *and* graph.** Plain RAG finds text *similar* to the question, which is weak for relationship questions ("what does X lead to?") whose answer is a chain spread across the graph. `/graphrag/ask` runs **two paths in parallel**: (1) vector search over Qdrant for descriptive passages; (2) an LLM extracts the entities the question mentions, resolves them to graph nodes, and traverses Neo4j for the surrounding relationship chains. Retrieved triples are **relevance-ranked** against the question (so a 2-hop neighbourhood doesn't flood the context), then both paths are merged into the LLM's context. On a relationship question this traces the real chain (`Proforma -> Size Order -> Delivery Out -> Sales Transaction`) that plain RAG misses — verifiable by toggling `use_graph`.

**Community summaries ("long documents").** GraphRAG answers *local* questions well but not *global* ones ("what are the main themes?"). `/graph/communities` runs **GDS Louvain** to find densely-connected clusters, then summarizes each — grounded in both its triples **and** retrieved source passages, so a thin cluster can't be described from the model's general priors. These per-theme summaries are the high-level documents that answer whole-corpus questions.

**Conservative entity resolution.** `/graph/resolve-entities` merges duplicate nodes, but the domain has near-identical names that are *different* things (`Size Order` vs `Sales Order`). So merges are **LLM-judged** (reasoning about meaning, not string similarity), explicitly warned that lookalike names are often distinct, and **dry-run by default** — it proposes merges for review before anything is changed, because a wrong merge is unrecoverable.

**Semantic chunking.** The default chunker embeds every sentence, finds where the *meaning* shifts (cosine distance between neighbours), and splits at the biggest topic jumps — adapting per file rather than using a fixed threshold. Switchable via `CHUNKING_STRATEGY`. Sentence embeddings run concurrently to stay fast on CPU-only Ollama.

**Content-hash deduplication.** The `file_id` is a SHA-256 hash of the file content (folded into a UUID). Qdrant point IDs are built from it, so re-ingesting the same file **overwrites in place** instead of duplicating. Edit the file -> hash changes -> treated as new.

**Two-layer anti-hallucination guard.** On `/ask`: (1) a **confidence guard** — if the best chunk's score is below `0.45`, the LLM is never called; (2) a **prompt guard** — the system prompt restricts the model to the provided context. Every answer carries its sources.

---

## n8n workflows

Exported workflow JSON lives in `n8n_workflows/`:

- **Ingest webhook** (`ingest_webhook.json`) — POST a file to a webhook; routes to `/ingest/pdf` or `/ingest/video` by file type.
- **Upload + Ingest + Agent** (`upload_ingest_minutes_agent_workflow.json`) — the main workflow. A form uploads a PDF/video -> ingest -> if it's a video, generate minutes and email them (Gmail). The **AI Agent** (Ollama chat model + windowed memory) has four tools: the **native Qdrant Vector Store** node for semantic search, plus HTTP tools for **GraphRAG** (`/graphrag/ask-simple`), **theme summaries** (`/graph/communities`), and **minutes-by-name** (`/minutes/by-name`, which emails via an internal webhook branch).
- **AI Agent (standalone)** (`ai_agent_workflow.json`) — an earlier chat-only version, kept for reference.

> **Note:** the agent was migrated from a custom `/agent/search` HTTP tool to the **native Qdrant Vector Store node**. That node is LangChain-based and reads a chunk's text from a configurable payload key (`Content Payload Key = text`) with metadata under a nested `metadata` key — so `qdrant_store` writes a nested `metadata` object (with a ready-made citation label) alongside the flat fields, keeping citations intact through the swap.

n8n runs in Docker, so it reaches the host API via `host.docker.internal:8000` (not `localhost`).

---

## Project structure

```
app/
  main.py                  # FastAPI entry point
  config.py                # all config from .env, no hardcoded values
  api/                     # routes for ingest, search, ask, agent, minutes,
                           #   documents, extract, graph, graphrag, communities,
                           #   resolve-entities
  services/                # ingestion, search, answer, minutes, extraction,
                           #   graph, graphrag, community, entity_resolution
  parsers/                 # pdf_parser, ocr_parser, video_parser,
                           #   audio_extractor, vision_client
  transcription/           # transcription_client (Whisper)
  chunking/                # simple_chunker + semantic_chunker
  extraction/              # entity_extractor (triples via gpt-4o-mini, JSON mode)
  embeddings/              # embedding_client
  vector_store/            # qdrant_store + graph_store (Qdrant / Neo4j gateways)
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
- **Cross-lingual retrieval** — `nomic-embed-text` is English-first. Measured: Arabic->Arabic retrieval is strong (~0.9), but an **English query against Arabic speech** retrieves weakly (the correct Arabic chunk can rank below unrelated content). Currently masked because vision-OCR indexes the English on-screen text. A 768-dim **multilingual** embedding model (e.g. `embeddinggemma`) in an aligned space fixes it; it's a re-embed of the existing chunks (no re-ingest), scoped but deferred.
- **Entity-merge path is built but untested** — resolution correctly proposes *no* merges on the current clean graph, so the merge code itself hasn't executed on real duplicates yet.
- **Community summaries can drift on tiny clusters** — mitigated by grounding in source passages and a `min_size` filter; a 2-node cluster still has little to describe.
- **Graph tooling is lightweight, not a framework** — GraphRAG + communities are built directly on Neo4j/Qdrant rather than adopting Microsoft GraphRAG / LightRAG, to stay explainable and reuse the existing pipeline. A framework would add hierarchical communities and scale; a candidate upgrade.

---

## What I learned building this

- Designing a pipeline around a **shared seam** so very different inputs (PDF, video frames, audio) reuse the same embed/store/search core.
- Treating the **context window** as a first-class constraint — RAG retrieval for questions, **map-reduce** for whole-transcript summarization.
- The difference between **retrieval and memory** in an agent, and why documents belong in the vector store, not chat memory.
- Practical **RAG guardrails** — confidence thresholds and prompt constraints — to stop confident-but-wrong answers.
- **Adapting under constraints** — swapping a gated/heavy vision model for an available one without touching the rest of the pipeline (isolated client module).
- Wiring **Docker networking** correctly (n8n reaching the host via `host.docker.internal`).
- Building a **knowledge graph** from unstructured transcripts — triple extraction with a controlled vocabulary, and why entity **normalization** matters or the graph fragments.
- **GraphRAG**: why relationship questions need graph traversal, not just vector similarity — and proving it with a `use_graph` on/off A/B on the same question.
- **Community detection** (GDS Louvain) to turn a graph into thematic "long documents," and grounding those summaries in source text to stop hallucination on thin clusters.
- **Measuring before optimizing** — testing cross-lingual retrieval directly (scores, not assumptions) before deciding whether a multilingual migration was worth the risk.
- Designing **destructive operations to be safe** — dry-run-by-default entity merges, LLM-judged rather than similarity-judged, because a wrong merge is unrecoverable.
