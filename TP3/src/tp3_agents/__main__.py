from __future__ import annotations

from .router import route_people


def _smoke() -> None:
    qs = [
        "¿Qué sabe hacer en Python?",
        "¿Qué experiencia en NLP tiene María García?",
        "Compará al alumno y a María García para un rol de Data Scientist",
    ]
    for q in qs:
        print(q, "->", route_people(q))


if __name__ == "__main__":
    _smoke()

