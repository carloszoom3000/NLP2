from __future__ import annotations

from groq import Groq

from .config import settings


def groq_client() -> Groq:
    s = settings()
    return Groq(api_key=s.groq_api_key)


def chat_completion(system: str, user: str) -> str:
    s = settings()
    client = groq_client()
    res = client.chat.completions.create(
        model=s.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (res.choices[0].message.content or "").strip()

