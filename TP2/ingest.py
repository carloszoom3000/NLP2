#!/usr/bin/env python3
"""
Carga CVs desde data/cvs/, calcula embeddings con Jina y sube vectores a Pinecone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pinecone import Pinecone

from rag_core import (
    DEFAULT_CVS_DIR,
    embed_texts,
    get_embedder,
    get_pinecone_index,
    index_name,
    load_cv_documents,
    pinecone_client,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingerir CVs en Pinecone")
    parser.add_argument(
        "--cvs-dir",
        type=Path,
        default=DEFAULT_CVS_DIR,
        help="Carpeta con archivos .txt de CVs",
    )
    args = parser.parse_args()

    docs = load_cv_documents(cvs_dir=args.cvs_dir)
    print(f"Documentos: {len(docs)} | índice Pinecone: {index_name()}")

    model = get_embedder()
    dim = model.get_sentence_embedding_dimension()
    texts = [t for _, _, t in docs]
    vectors = embed_texts(model, texts)

    pc: Pinecone = pinecone_client()
    index = get_pinecone_index(pc, dimension=dim)

    if len(docs) != len(vectors):
        raise RuntimeError("Cantidad de documentos y vectores no coincide.")
    records = []
    for (doc_id, fname, text), vec in zip(docs, vectors):
        records.append(
            {
                "id": doc_id,
                "values": vec,
                "metadata": {
                    "filename": fname,
                    "text": text[:39000],
                },
            }
        )

    index.upsert(vectors=records)
    print(f"Upsert OK: {len(records)} vectores (dim={dim}).")


if __name__ == "__main__":
    main()
