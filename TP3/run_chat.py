#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Emitted on urllib3 import when Python is built against LibreSSL (common on macOS).
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tp3_agents import answer_multi, route_people
from tp3_agents.embeddings import get_embedder
from tp3_agents.pinecone_store import get_index


def _argv_without_shell_comment(argv: list[str]) -> list[str]:
    """Drop `# ...` tail when the shell passes it (e.g. Windows cmd has no `#` comments)."""
    try:
        i = argv.index("#")
    except ValueError:
        return argv
    return argv[:i]


def main() -> None:
    parser = argparse.ArgumentParser(description="Router multi-agente (1 por persona) + Pinecone + Groq")
    parser.add_argument("question", type=str, help="Pregunta en lenguaje natural")
    parser.add_argument("--top-k", type=int, default=6, help="Vecinos por persona a recuperar")
    args = parser.parse_args(_argv_without_shell_comment(sys.argv[1:]))

    # Index (asegura dimensión consistente con el embedder)
    model = get_embedder()
    dim = model.get_sentence_embedding_dimension()
    index = get_index(dimension=dim)

    people = route_people(args.question)
    print(f"Route -> agents/person_ids: {people}")

    final = answer_multi(index, args.question, people)
    print()
    print(final)


if __name__ == "__main__":
    main()

