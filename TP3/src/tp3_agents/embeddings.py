from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


_ROOT = Path(__file__).resolve().parents[2]  # TP3/
_HF_CACHE = _ROOT / ".hf_cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

# Modelo liviano y consistente con TP2 (embeddings normalizados -> coseno = dot)
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v2-small-en"

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_ID, trust_remote_code=True)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Devuelve embeddings L2-normalizados.
    Con normalización: cosine(a,b) == dot(a,b)
    """
    model = get_embedder()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 8,
    )
    return vecs.astype(np.float32, copy=False)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

