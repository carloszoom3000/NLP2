# TP3 — Sistema de agentes (1 por persona) con Pinecone + Groq

## Objetivo
- Cargar **2 CVs** (texto) → generar **embeddings** → subir vectores a **Pinecone**
- Hacer preguntas y, mediante un **router (“conditional edge”)** basado en `re.match`, decidir qué agente(s) usar:
  - Si **no se nombra a nadie** en la query → **Agente del alumno**
  - Si se consulta por **más de un CV** → traer contexto de **cada uno** y responder acorde
- En la demo se calcula **similitud coseno manual** (producto punto con embeddings normalizados) para elegir el vector más cercano.

## Setup
Las variables se leen desde la carpeta **TP3** (no importa desde qué directorio ejecutes `python`): primero `.env.example`, luego `.env` (si existe, tiene prioridad).

1) Crear tu `.env` a partir de `.env.example` y pegar tus claves (recomendado para no commitear secretos):

```bash
cp .env.example .env
# editá .env con tus PINECONE_API_KEY y GROQ_API_KEY
```

2) Instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cargar CVs (2 personas)
Los CVs se leen desde `data/cvs/*.txt`. Ya hay 2 ejemplos:
- `data/cvs/alumno.txt`
- `data/cvs/maria_garcia.txt`

Podés reemplazarlos por tus CVs reales manteniendo el nombre del archivo (se usa como `person_id`).

## Ingestar (embeddings → Pinecone)

```bash
python run_ingest.py
```

## Probar preguntas (router multi-agente + coseno manual)

```bash
python run_chat.py "¿Qué experiencia en NLP tiene María García?"
python run_chat.py "Compará al alumno y a María García para un rol de Data Scientist"
python run_chat.py "¿Qué sabe hacer en Python?"
# (sin mencionar a nadie, el router usa el CV del alumno por defecto)
```

## Cómo funciona el “conditional edge”
El router está en `src/tp3_agents/router.py` y usa patrones con `re.match(r\".*\\bNOMBRE\\b.*\", query, re.I)` para decidir:
- `['alumno']` por defecto
- `['maria_garcia']` si se menciona
- `['alumno', 'maria_garcia']` si se mencionan ambos

