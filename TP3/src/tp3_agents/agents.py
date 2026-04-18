from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import cosine_sim, embed_texts
from .llm import chat_completion
from .pinecone_store import Match, query_person
from .router import PEOPLE


@dataclass(frozen=True)
class AgentAnswer:
    person_id: str
    display_name: str
    nearest_chunk_id: str | None
    nearest_cosine: float | None
    context: str
    answer: str


def _build_context(matches: list[Match], *, max_chars: int = 3500) -> str:
    parts: list[str] = []
    total = 0
    for m in matches:
        txt = (m.metadata.get("text") or "").strip()
        if not txt:
            continue
        block = f"[chunk {m.id} | pinecone_score={m.score:.4f}]\n{txt}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def _manual_nearest_by_cosine(query_vec: np.ndarray, matches: list[Match]) -> tuple[str, float] | tuple[None, None]:
    best_id: str | None = None
    best_score: float | None = None
    for m in matches:
        if not m.values:
            continue
        v = np.asarray(m.values, dtype=np.float32)
        score = cosine_sim(query_vec, v)
        if best_score is None or score > best_score:
            best_score = score
            best_id = m.id
    return best_id, best_score


def answer_for_person(index, question: str, person_id: str, *, top_k: int = 6) -> AgentAnswer:
    info = PEOPLE.get(person_id, {})
    display = str(info.get("display") or person_id)

    qvec = embed_texts([question])[0]
    matches = query_person(
        index,
        query_vector=qvec.tolist(),
        person_id=person_id,
        top_k=top_k,
        include_values=True,
    )

    nearest_id, nearest_cos = _manual_nearest_by_cosine(qvec, matches)
    context = _build_context(matches)

    system = (
        "Sos un asistente de RRHH/Recruiting. Respondé SOLO usando el contexto del CV provisto. "
        "Si falta info, decilo explícitamente."
    )
    user = (
        f"Persona objetivo: {display} (person_id={person_id})\n\n"
        f"Contexto (extractos del CV):\n{context}\n\n"
        f"Pregunta:\n{question}\n\n"
        "Respuesta (clara y concisa):"
    )
    ans = chat_completion(system=system, user=user)

    return AgentAnswer(
        person_id=person_id,
        display_name=display,
        nearest_chunk_id=nearest_id,
        nearest_cosine=nearest_cos,
        context=context,
        answer=ans,
    )


def answer_multi(index, question: str, person_ids: list[str]) -> str:
    answers = [answer_for_person(index, question, pid) for pid in person_ids]

    if len(answers) == 1:
        a = answers[0]
        header = (
            f"[Agente: {a.display_name}] "
            f"(nearest_chunk={a.nearest_chunk_id}, cosine={a.nearest_cosine})"
        )
        return f"{header}\n{a.answer}".strip()

    # Si hay múltiples CVs, devolvemos una síntesis/comparación con Groq usando las respuestas parciales.
    system = (
        "Sos un asistente que combina resultados de múltiples agentes. "
        "No inventes datos: usá SOLO lo que te dieron los agentes."
    )
    blocks: list[str] = []
    for a in answers:
        blocks.append(
            f"---\nAGENTE: {a.display_name} (person_id={a.person_id})\n"
            f"nearest_chunk={a.nearest_chunk_id} cosine={a.nearest_cosine}\n"
            f"RESPUESTA:\n{a.answer}\n"
        )
    user = (
        f"Pregunta original:\n{question}\n\n"
        f"Respuestas de agentes:\n{''.join(blocks)}\n\n"
        "Generá una respuesta final que compare/integre cuando corresponda. "
        "Si la pregunta pide comparar perfiles, devolvé un breve resumen por persona y una conclusión."
    )
    return chat_completion(system=system, user=user)

