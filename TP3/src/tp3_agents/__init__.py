from __future__ import annotations

# Mantener este módulo liviano: no importar torch/sentence-transformers al importar el paquete.
from .router import route_people


def answer_multi(index, question: str, person_ids: list[str]) -> str:
    from .agents import answer_multi as _answer_multi

    return _answer_multi(index, question, person_ids)


__all__ = ["answer_multi", "route_people"]

