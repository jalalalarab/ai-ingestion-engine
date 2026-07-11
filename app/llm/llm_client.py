"""
LLM client — wraps Ollama's /api/chat endpoint.

One job: given a system prompt + user prompt, return the assistant's text answer.

Works with both local models (via localhost:11434) and Ollama Cloud models
(model names ending in -cloud). Same endpoint, same API — routing happens
inside Ollama based on the model name.

For reasoning models (like gpt-oss:*), the API may return a separate `thinking`
field. We ignore it and return only `content` — clean answers, no chain-of-thought.
"""
import httpx
from app.config import settings


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """
    Ask the LLM to respond to a user_prompt under a given system_prompt.

    Returns:
        The plain-text assistant answer (thinking output stripped).

    Raises:
        RuntimeError: if the response is missing the expected content field.
        httpx.HTTPError: if the Ollama call itself fails.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,  # get one complete response, not a token stream
        "think": False,   # ask reasoning models to skip visible thinking when possible
    }

    with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    message = data.get("message", {})
    content = message.get("content", "")
    if not content:
        # Some models put reasoning-only output; fall back gracefully.
        raise RuntimeError(
            f"LLM returned empty content. Full message: {message}"
        )
    return content.strip()