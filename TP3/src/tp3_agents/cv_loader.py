from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CvChunk:
    chunk_id: str
    person_id: str
    filename: str
    text: str


def _chunk_text(text: str, *, max_chars: int = 900, overlap: int = 120) -> list[str]:
    """
    Chunking simple por caracteres con solapamiento.
    Suficiente para el TP: mantiene contexto sin complicar.
    """
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return [t]
    chunks: list[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return chunks


def load_cv_chunks(cvs_dir: Path) -> list[CvChunk]:
    if not cvs_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de CVs: {cvs_dir}")

    out: list[CvChunk] = []
    for path in sorted(cvs_dir.glob("*.txt")):
        person_id = path.stem.strip()
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        pieces = _chunk_text(raw)
        for i, piece in enumerate(pieces):
            out.append(
                CvChunk(
                    chunk_id=f"{person_id}#{i}",
                    person_id=person_id,
                    filename=path.name,
                    text=piece,
                )
            )

    if not out:
        raise RuntimeError(f"No hay archivos .txt en {cvs_dir}")
    return out

