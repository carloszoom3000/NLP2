from __future__ import annotations

import re


# 1 agente por persona (person_id == nombre de archivo sin extensión)
PEOPLE: dict[str, dict[str, object]] = {
    "alumno": {
        "display": "Alumno",
        "patterns": [
            r".*\balumno\b.*",
            r".*\bcarlos\b.*",
            r".*\bvillalobos\b.*",
        ],
    },
    "maria_garcia": {
        "display": "María García",
        "patterns": [
            r".*\bmar[ií]a\b.*",
            r".*\bgarc[ií]a\b.*",
            r".*\bmar[ií]a\s+garc[ií]a\b.*",
        ],
    },
}


def route_people(query: str) -> list[str]:
    """
    Conditional Edge (decisión) usando re.match:
    - Si no se menciona a nadie -> ['alumno']
    - Si se mencionan múltiples -> lista de person_id correspondientes
    """
    q = (query or "").strip()
    if not q:
        return ["alumno"]

    selected: list[str] = []
    for person_id, info in PEOPLE.items():
        patterns = list(info.get("patterns") or [])
        for pat in patterns:
            if re.match(pat, q, flags=re.IGNORECASE):
                selected.append(person_id)
                break

    # default: alumno
    if not selected:
        return ["alumno"]

    # dedupe preservando orden
    seen: set[str] = set()
    out: list[str] = []
    for p in selected:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

