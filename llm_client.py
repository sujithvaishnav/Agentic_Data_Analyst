"""
Thin wrapper around the Groq OpenAI-compatible API.
"""

from __future__ import annotations
import json
from openai import OpenAI
from config.settings import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


def _client() -> OpenAI:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to .env or export as env var."
        )
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def chat(
    system: str,
    user: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Single-turn chat completion. Returns the assistant message string."""
    client = _client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def chat_json(system: str, user: str, **kwargs) -> dict:
    """
    Same as chat() but parses the response as JSON.
    The system prompt must instruct the LLM to reply ONLY with valid JSON.
    """
    raw = chat(system, user, **kwargs)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
