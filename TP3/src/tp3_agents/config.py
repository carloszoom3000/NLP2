from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Raíz del proyecto TP3 (donde están run_ingest.py, data/, .env)
_TP3_ROOT = Path(__file__).resolve().parent.parent.parent

# Sin ruta, load_dotenv() solo busca .env en el cwd — falla si corrés desde otra carpeta.
# Orden: primero plantilla, luego .env sobrescribe (override=True).
_env_example = _TP3_ROOT / ".env.example"
_env_local = _TP3_ROOT / ".env"
if _env_example.is_file():
    load_dotenv(_env_example, override=False)
if _env_local.is_file():
    load_dotenv(_env_local, override=True)


def _env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Falta variable de entorno requerida: {key}")
    val = str(val).strip()
    if not val:
        raise RuntimeError(f"Falta variable de entorno requerida: {key}")
    return val


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    groq_api_key: str
    groq_model: str


def settings() -> Settings:
    return Settings(
        pinecone_api_key=_env("PINECONE_API_KEY"),
        pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME", "nlp2-tp3-cvs").strip()
        or "nlp2-tp3-cvs",
        pinecone_cloud=os.environ.get("PINECONE_CLOUD", "aws").strip() or "aws",
        pinecone_region=os.environ.get("PINECONE_REGION", "us-east-1").strip() or "us-east-1",
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        or "llama-3.3-70b-versatile",
    )

