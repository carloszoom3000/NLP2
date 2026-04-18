from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from .config import settings


@dataclass(frozen=True)
class Match:
    id: str
    score: float
    values: list[float] | None
    metadata: dict[str, Any]


def pinecone_client() -> Pinecone:
    s = settings()
    return Pinecone(api_key=s.pinecone_api_key)


def _serverless_spec() -> ServerlessSpec:
    s = settings()
    return ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region)


def get_index(dimension: int):
    s = settings()
    pc = pinecone_client()
    existing = set(pc.list_indexes().names())
    if s.pinecone_index_name not in existing:
        pc.create_index(
            name=s.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=_serverless_spec(),
        )
    return pc.Index(s.pinecone_index_name)


def upsert_records(index, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    index.upsert(vectors=records)


def query_person(
    index,
    query_vector: list[float],
    person_id: str,
    top_k: int = 5,
    include_values: bool = True,
) -> list[Match]:
    res = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        include_values=include_values,
        filter={"person_id": {"$eq": person_id}},
    )
    out: list[Match] = []
    for m in list(res.matches or []):
        out.append(
            Match(
                id=m.id,
                score=float(m.score),
                values=list(m.values) if getattr(m, "values", None) is not None else None,
                metadata=dict(m.metadata or {}),
            )
        )
    return out

