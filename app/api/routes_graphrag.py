"""
GraphRAG route — POST /graphrag/ask.

Answer a question using both retrieval paths (vector + graph). The `use_graph`
flag lets you run the SAME question as GraphRAG (true) or plain RAG (false) to
compare answers — the whole point of Phase D is seeing where the graph helps.

Thin route: delegates to the graphrag service.
"""
from pydantic import BaseModel
from fastapi import APIRouter

from app.services.graphrag_service import answer_question

router = APIRouter(prefix="/graphrag", tags=["graphrag"])


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    hops: int = 2
    use_graph: bool = True  # false = plain RAG, for comparison


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    vector_chunks_used: int
    graph_entities_resolved: list[str]
    graph_triples_used: list[Triple]


@router.post("/ask", response_model=AskResponse)
async def graphrag_ask_endpoint(req: AskRequest) -> AskResponse:
    """
    Answer a question with GraphRAG (vector + graph) or plain RAG.

    Set use_graph=false to get the plain-RAG answer for the same question, so you
    can compare them side by side on relationship-heavy questions.
    """
    result = answer_question(
        req.question,
        top_k=req.top_k,
        hops=req.hops,
        use_graph=req.use_graph,
    )
    return AskResponse(**result)


@router.post("/ask-simple", response_model=AskResponse)
async def graphrag_ask_simple_endpoint(
    question: str,
    top_k: int = 5,
    hops: int = 2,
    use_graph: bool = True,
) -> AskResponse:
    """
    Query-param version of /ask — built for the n8n agent tool.

    Same GraphRAG answer, but the question arrives as ?question=... instead of in
    a JSON body, because n8n's HTTP Request Tool fills QUERY parameters reliably
    ("By Model") while JSON-body placeholders proved fragile in this n8n version.
    """
    result = answer_question(
        question,
        top_k=top_k,
        hops=hops,
        use_graph=use_graph,
    )
    return AskResponse(**result)
