"""
Answer service — the RAG glue.

Steps:
  1. Retrieve top chunks via search_service (semantic search).
  2. If no chunk clears the confidence threshold → return "not found".
  3. Otherwise, build a prompt with the retrieved chunks as context.
  4. Call the LLM.
  5. Return answer + structured sources for citation.

The system prompt is the anti-hallucination guard: the model is instructed
to answer ONLY from the provided context and to say "not found" if it can't.
"""
from dataclasses import dataclass, field
from app.services.search_service import search_chunks
from app.llm.llm_client import generate_answer


# Minimum cosine similarity for a chunk to count as "relevant."
# Below this, we treat retrieval as failed and skip the LLM entirely.
# Based on your Phase 2 tests: relevant queries scored 0.55+, unrelated ~0.37.
# 0.45 is a defensible middle ground.
MIN_SCORE_FOR_CONTEXT = 0.45


SYSTEM_PROMPT = """You are a precise question-answering assistant for a document search system.

STRICT RULES:
1. Answer ONLY using the information in the "CONTEXT" section below.
2. If the context does not contain the answer, respond exactly: "I could not find the answer in the provided documents."
3. Do NOT use outside knowledge. Do NOT guess.
4. Keep answers short and direct — 1-3 sentences.
5. When citing, refer to the source's page number when available.
6. If multiple context passages agree, synthesize them; if they disagree, note the disagreement.
"""


NOT_FOUND_MESSAGE = "I could not find the answer in the provided documents."


@dataclass
class Source:
    file_id: str | None
    file_name: str | None
    source_type: str | None
    page_number: int | None            # PDF only; None for video
    timestamp_seconds: int | None      # video only; None for PDF
    frame_number: int | None           # video only; None for PDF
    chunk_index: int | None
    score: float
    label: str  # human-readable, e.g. "Day1_Study_Guide.pdf — p.3" or "clip.mp4 — 04:32"


@dataclass
class AnswerResponse:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)


def _format_timestamp(seconds: int) -> str:
    """Turn a whole-second offset into MM:SS (or H:MM:SS past an hour)."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _label_for(hit: dict) -> str:
    """Human-readable citation label — e.g. 'file.pdf — p.5' or 'video.mp4 — 04:32'."""
    name = hit.get("file_name") or "unknown"
    if hit.get("page_number") is not None:
        return f"{name} — p.{hit['page_number']}"
    # Video branch — cite the moment in the clip when we have a timestamp.
    if hit.get("timestamp_seconds") is not None:
        return f"{name} — {_format_timestamp(hit['timestamp_seconds'])}"
    return name


def _build_user_prompt(question: str, hits: list[dict]) -> str:
    """Assemble the retrieved chunks + question into a single prompt string."""
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        label = _label_for(hit)
        context_blocks.append(f"[Passage {i} — {label}]\n{hit['text']}")
    context = "\n\n".join(context_blocks)

    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the CONTEXT above."
    )


def answer_question(
    question: str,
    top_k: int = 5,
    file_id: str | None = None,
    source_type: str | None = None,
) -> AnswerResponse:
    """
    Full RAG: retrieve → check confidence → prompt LLM → return answer + sources.
    """
    # Guard: empty question
    if not question or not question.strip():
        return AnswerResponse(
            question=question,
            answer="Please provide a question.",
            sources=[],
        )

    # 1. Retrieve
    hits = search_chunks(
        query=question,
        top_k=top_k,
        file_id=file_id,
        source_type=source_type,
    )

    # 2. Confidence guard — no usable context, refuse to hallucinate
    if not hits or hits[0]["score"] < MIN_SCORE_FOR_CONTEXT:
        return AnswerResponse(
            question=question,
            answer=NOT_FOUND_MESSAGE,
            sources=[],
        )

    # 3 + 4. Build prompt and call the LLM
    user_prompt = _build_user_prompt(question, hits)
    llm_answer = generate_answer(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # 5. Package sources (only chunks above the threshold — noise below is dropped)
    sources = [
        Source(
            file_id=hit.get("file_id"),
            file_name=hit.get("file_name"),
            source_type=hit.get("source_type"),
            page_number=hit.get("page_number"),
            timestamp_seconds=hit.get("timestamp_seconds"),
            frame_number=hit.get("frame_number"),
            chunk_index=hit.get("chunk_index"),
            score=hit["score"],
            label=_label_for(hit),
        )
        for hit in hits
        if hit["score"] >= MIN_SCORE_FOR_CONTEXT
    ]

    return AnswerResponse(
        question=question,
        answer=llm_answer,
        sources=sources,
    )