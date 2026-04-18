"""
Utilidades compartidas: embeddings (Jina v2 small) y cliente Pinecone.
"""
from __future__ import annotations

import os
from pathlib import Path

# Caché de Hugging Face dentro del proyecto (evita depender de ~/.cache en CI o permisos raros)
_ROOT = Path(__file__).resolve().parent
_HF_CACHE = _ROOT / ".hf_cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v2-small-en"
DEFAULT_CVS_DIR = Path(__file__).resolve().parent / "data" / "cvs"

_embedder: SentenceTransformer | None = None


def _pinecone_api_key() -> str:
    key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Definí PINECONE_API_KEY en el archivo .env")
    return key


def pinecone_client() -> Pinecone:
    return Pinecone(api_key=_pinecone_api_key())


def index_name() -> str:
    return os.environ.get("PINECONE_INDEX_NAME", "nlp2-tp2-cvs").strip()


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_ID, trust_remote_code=True)
    return _embedder


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Normaliza L2 como recomienda el modelo Jina para similitud coseno vía producto punto."""
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 8,
    )
    return vectors.tolist()


def _serverless_spec() -> ServerlessSpec:
    cloud = os.environ.get("PINECONE_CLOUD", "aws").strip()
    region = os.environ.get("PINECONE_REGION", "us-east-1").strip()
    return ServerlessSpec(cloud=cloud, region=region)


def get_pinecone_index(pc: Pinecone, dimension: int):
    name = index_name()
    existing = set(pc.list_indexes().names())
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=_serverless_spec(),
        )
    return pc.Index(name)


def load_cv_documents(cvs_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """
    Devuelve lista de (id, nombre_archivo, texto) por cada .txt en el directorio.
    """
    base = cvs_dir or DEFAULT_CVS_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"No existe el directorio de CVs: {base}")
    out: list[tuple[str, str, str]] = []
    for path in sorted(base.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_id = path.stem
        out.append((doc_id, path.name, text))
    if not out:
        raise RuntimeError(f"No hay archivos .txt en {base}")
    return out
