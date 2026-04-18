#!/usr/bin/env python3
"""
Prueba de recuperación: embedding de la pregunta + consulta a Pinecone (métrica coseno).
Imprime el documento más cercano y la puntuación devuelta por Pinecone.
"""
from __future__ import annotations

import argparse

from rag_core import embed_texts, get_embedder, get_pinecone_index, pinecone_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consulta el vector más cercano (similitud coseno en Pinecone)"
    )
    parser.add_argument("question", type=str, help="Pregunta en lenguaje natural")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Cantidad de vecinos a listar",
    )
    args = parser.parse_args()

    model = get_embedder()
    dim = model.get_sentence_embedding_dimension()
    pc = pinecone_client()
    index = get_pinecone_index(pc, dimension=dim)

    qvec = embed_texts(model, [args.question])[0]
    res = index.query(
        vector=qvec,
        top_k=args.top_k,
        include_metadata=True,
    )

    matches = list(res.matches or [])

    print(f"Pregunta: {args.question!r}\n")
    for i, m in enumerate(matches, start=1):
        meta = dict(m.metadata or {})
        fname = meta.get("filename", "?")
        snippet = (meta.get("text") or "")[:400].replace("\n", " ")
        score = m.score
        print(f"--- Rango {i} ---")
        print(f"id: {m.id} | score (Pinecone/coseno): {score}")
        print(f"archivo: {fname}")
        print(f"texto (extracto): {snippet}...")
        print()


if __name__ == "__main__":
    main()
