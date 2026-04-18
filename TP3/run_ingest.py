#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tp3_agents.cv_loader import load_cv_chunks
from tp3_agents.embeddings import embed_texts, get_embedder
from tp3_agents.pinecone_store import get_index, upsert_records


def main() -> None:
    root = Path(__file__).resolve().parent
    cvs_dir = root / "data" / "cvs"

    chunks = load_cv_chunks(cvs_dir)
    texts = [c.text for c in chunks]

    model = get_embedder()
    dim = model.get_sentence_embedding_dimension()
    index = get_index(dimension=dim)

    vecs = embed_texts(texts)

    records = []
    for c, v in zip(chunks, vecs):
        records.append(
            {
                "id": c.chunk_id,
                "values": v.tolist(),
                "metadata": {
                    "person_id": c.person_id,
                    "filename": c.filename,
                    "text": c.text[:39000],
                },
            }
        )

    upsert_records(index, records)
    people = sorted({c.person_id for c in chunks})
    print(f"Upsert OK: {len(records)} chunks (dim={dim}) | people={people}")


if __name__ == "__main__":
    main()

