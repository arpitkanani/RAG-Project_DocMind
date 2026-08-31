# DocMind

A local-first, full-stack Retrieval-Augmented Generation (RAG) application. Upload documents
(PDF, DOCX, TXT, CSV, MD, XLSX) or a YouTube URL, and ask questions or request summaries
grounded in that content, with multi-turn conversation memory and cited sources.

Built with **FastAPI** (backend + HTML frontend), **LangChain** (RAG orchestration),
**Qdrant** (vector search), **PostgreSQL** (users, sessions, chat history), a local embedding
model, and a fast hosted LLM (Groq or Google) for generation.

---

## What it does

- **Ingests documents** of several types (PDF/DOCX/TXT/CSV/MD/XLSX, up to 20MB each, 5 files at
  once) or a YouTube URL (transcript-based), chunks them, embeds them locally, and stores them
  in a per-document Qdrant collection.
- **Answers questions** grounded only in the retrieved content: semantic search in Qdrant,
  followed by a lexical/semantic-blended reranking pass, followed by generation with an
  anti-hedging prompt that refuses to answer beyond what was retrieved.
- **Summarizes** documents or YouTube videos on request, using a wider "full context" retrieval
  mode instead of the query-scoped search path.
- **Remembers conversations** per session, across multiple documents attached to that session
  at once, with older turns automatically compressed into an LLM-generated summary so long
  conversations don't balloon token usage.
- **Cites its sources** in every RAG answer, tied back to the specific document/page/section a
  claim came from.

---

## Architecture

```
Browser (templates/static/js/app_new.js)
        |
        v
FastAPI (app.py)  ---- X-API-Key header ---->  PostgreSQL (users, sessions, messages, attachments)
        |
        +--> IngestionPipeline  --> local embedding model --> Qdrant (per-doc collection)
        |
        +--> QAPipeline
                |
                +--> MemoryManager        (Postgres: recent turns verbatim + older turns summarized)
                +--> Retriever            (Qdrant search -> lexical/semantic rerank -> threshold filter)
                +--> LangChain QA chain   (prompt -> Groq/Google LLM -> answer)
```

### Why PostgreSQL (not just files or an in-memory store)

Chat sessions, messages, attachments, and user accounts are relational, multi-user data with
real query patterns behind them — "give me this user's sessions ordered by recency," "prune
messages older than N days," "look up a user by API key on every request." Postgres gives that
for free (indexes, foreign keys with cascade delete, transactional writes) instead of
hand-rolling it against flat JSON files, which is what an earlier version of this project did
(`session_state_{id}.json`, `chat_history_{id}.json` per session). The schema
(`database/init.sql`) explicitly mirrors that earlier JSON structure table-for-table, so the
switch was a storage-layer change, not a data-model rewrite. It also unlocks straightforward
multi-user support and safe concurrent access, neither of which flat files handle well once
more than one session is being written at a time.

### Why API-key auth (not a login/password flow)

There's no username/password, no session cookie, no login page. Every request instead carries
an `X-API-Key` header, which the backend hashes (SHA-256) and looks up against the `users`
table (`src/auth.py`). Keys are minted once, offline, with `python seed_user.py "Name"` — the
plaintext key is shown exactly once and only its hash is ever stored.

This is a deliberate trade for a project of this shape: it's typically self-hosted for a person
or a small team, not a public multi-tenant SaaS product. A full login system (password
hashing + reset flows + session/cookie management + CSRF handling) is a lot of surface area
and a lot of things to get wrong security-wise for a benefit — "users can set their own
password" — that doesn't matter much when accounts are provisioned by whoever runs the server.
An API key is simpler to reason about, works identically for a browser client and a future
CLI/API client, and is trivial to revoke (`is_active = false`) without needing password-reset
infrastructure at all. If this ever needs public self-serve signup, that's a real, separate
feature to add on top — not a sign the current approach is wrong for what it's serving today.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend + frontend | FastAPI, Jinja-style HTML templates, vanilla JS (`app_new.js`) |
| Vector store | Qdrant |
| Relational store | PostgreSQL |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally (CPU) |
| LLM | Groq (`llama-3.1-8b-instant`) or Google Gemini — config-switchable |
| Orchestration | LangChain (LCEL chains); LangGraph is being introduced on a separate branch |
| Reranking | Custom lexical/semantic-blended reranker (no external reranker model) |
| Auth | Hashed API keys, no login UI |

---

## Project layout

```
app.py                     FastAPI app: all routes, startup, request/response models
config/config.yaml         Every tunable: embedding model, LLM provider/params, retriever
                            thresholds, chunking, upload limits, YouTube limits, Postgres conn
database/init.sql          Postgres schema (users, sessions, messages, attachments)
seed_user.py                One-off script to create a user + issue an API key
src/
  auth.py                  X-API-Key -> user_id dependency
  logger.py, exception.py  Shared logging + custom exception types
  components/
    embedder.py            Local embedding model wrapper
    vector_store.py        Qdrant client/collection management
    retriever.py           Search + rerank + threshold/margin filtering + full-context mode
    memory_manager.py      Postgres-backed session/message/summary handling
    document_loader.py, text_splitter.py
  chains/qa_chain.py        Prompt, LLM factory, sanitizer, citation builder, the LCEL chain
  pipelines/
    ingestion_pipeline.py   File/YouTube -> chunks -> embeddings -> Qdrant
    qa_pipeline.py           Orchestrates a /query request end to end
  utils/
    job_manager.py           Background job tracking for uploads
    rate_limiter.py          Proactive LLM rate-limit handling
    file_helper.py, youtube_helper.py
templates/                  Frontend HTML + static/js/app_new.js
tests/, evaluation/          Test suite and RAG evaluation scripts
```

---

## Setup

### 1. Prerequisites

- Python 3.10
- A running **Qdrant** instance (default expected at `http://localhost:6333`) — easiest via
  Docker: `docker run -p 6333:6333 qdrant/qdrant`
- A running **PostgreSQL** instance
- A Groq API key (default LLM provider) and/or a Google API key

### 2. Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key         # only needed if config.yaml's llm.provider is "google"
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=docmind
POSTGRES_PASSWORD=your_password
POSTGRES_DB=docmind
```

### 3. Python environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows — use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

### 4. Database

Create the database, then load the schema:

```bash
createdb docmind
psql -d docmind -f database/init.sql
```

### 5. Create a user (API key)

```bash
python seed_user.py "Your Name"
```

This prints a `user_id` and a plaintext API key (`dk_live_...`) — copy the key now, it's shown
once. This is what the frontend/any client sends as `X-API-Key` on every request.

### 6. Config

`config/config.yaml` controls everything else: which LLM provider/model, retrieval thresholds
(`retriever.k`, `score_threshold`, `collection_margin`), chunking, upload limits, and YouTube
languages. Defaults are reasonable to start; nothing here needs to change to get running.

### 7. Run

```bash
uvicorn app:app --reload
```

Then open:

- `http://localhost:8000/` or `/app` — the chat UI
- `http://localhost:8000/landing` — landing page
- `http://localhost:8000/health` — health check (no auth required)

All other routes require the `X-API-Key` header from step 5.

---

## API surface (for the frontend or any external client)

| Method | Route | Purpose |
|---|---|---|
| POST | `/upload` | Upload one or more documents for ingestion |
| GET | `/upload/status/{job_id}` | Poll background ingestion job status |
| POST | `/youtube` | Ingest a YouTube URL by transcript |
| POST | `/query` | Ask a question / request a summary |
| GET | `/collections` | List a user's Qdrant collections |
| DELETE | `/collections/{collection_name}` | Delete a collection |
| POST | `/sessions`, `/sessions/new` | Create a chat session |
| GET | `/sessions` | List a user's sessions |
| GET | `/sessions/{session_id}` | Get one session's history |
| DELETE | `/sessions/{session_id}` | Delete a session |
| DELETE | `/sessions/{session_id}/attachments/{collection_name}` | Detach a document from a session |
| DELETE | `/memory` | Wipe all of a user's sessions/collections |
| GET | `/health` | Liveness check |

---

## Note on the `langgraph` branch

`requirements.txt` already includes `langgraph`/`langgraph-checkpoint`/`langgraph-sdk` —
orchestration is being migrated from a single LangChain LCEL chain to a LangGraph `StateGraph`
(retrieve → refine → generate, with token-by-token streaming to the frontend) on a separate
branch. Qdrant's collection structure and Postgres's schema are unaffected by that migration;
only `/query`'s internal orchestration and response streaming are changing.
