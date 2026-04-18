#!/usr/bin/env python3
"""
Chatbot RAG: recupera CVs relevantes en Pinecone y genera respuesta con Groq.
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from groq import Groq

from rag_core import embed_texts, get_embedder, get_pinecone_index, pinecone_client

load_dotenv()

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def groq_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Definí GROQ_API_KEY en el archivo .env")
    return Groq(api_key=key)


def retrieve_context(question: str, top_k: int) -> tuple[str, list[dict]]:
    model = get_embedder()
    dim = model.get_sentence_embedding_dimension()
    pc = pinecone_client()
    index = get_pinecone_index(pc, dimension=dim)
    qvec = embed_texts(model, [question])[0]
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
    matches = list(res.matches or [])
    blocks = []
    for m in matches:
        meta = dict(m.metadata or {})
        text = (meta.get("text") or "").strip()
        if not text:
            continue
        fname = meta.get("filename", m.id)
        blocks.append(f"### Documento: {fname}\n{text}")
    context = "\n\n".join(blocks) if blocks else "(sin contexto recuperado)"
    return context, matches


def answer_question(
    question: str,
    *,
    top_k: int,
    model_id: str,
) -> str:
    context, _matches = retrieve_context(question, top_k=top_k)
    client = groq_client()
    system = (
        "Sos un asistente que responde en español usando SOLO la información del "
        "contexto de currículums que recibís. Si algo no está en el contexto, decilo "
        "claramente. Sé conciso y menciona nombres o archivos cuando ayude."
    )
    user = f"Contexto (currículums):\n{context}\n\nPregunta: {question}"
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    choice = completion.choices[0].message
    return (choice.content or "").strip()


def run_interactive(top_k: int, model_id: str) -> None:
    print("Chat RAG sobre CVs (vacío o 'salir' para terminar).")
    while True:
        try:
            line = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in {"salir", "exit", "quit"}:
            break
        try:
            reply = answer_question(line, top_k=top_k, model_id=model_id)
        except Exception as e:  # noqa: BLE001 — mostrar error al usuario
            print(f"Error: {e}", file=sys.stderr)
            continue
        print(f"Bot: {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatbot RAG con Pinecone + Groq")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Una pregunta (si se omite, modo interactivo)",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Chunks/CVs a recuperar")
    parser.add_argument("--model", type=str, default=DEFAULT_GROQ_MODEL, help="Modelo Groq")
    args = parser.parse_args()

    if args.question:
        text = answer_question(args.question.strip(), top_k=args.top_k, model_id=args.model)
        print(text)
    else:
        run_interactive(top_k=args.top_k, model_id=args.model)


if __name__ == "__main__":
    main()
