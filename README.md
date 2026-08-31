# DocMind

**Local-first Retrieval-Augmented Generation (RAG) application.** Upload documents or a
YouTube URL, then ask questions or request summaries grounded in that content — with
multi-turn conversation memory and cited sources.

Built with **FastAPI** (backend + frontend), **LangChain** (RAG orchestration), **Qdrant**
(vector search), **PostgreSQL** (users, sessions, chat history), a local embedding model, and
a hosted LLM (Groq or Google) for generation.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Tech Stack](#tech-stack)
- [Project Layout](#project-layout)
- [Setup](#setup)
- [API Reference](#api-reference)
- [Roadmap](#roadmap)

---

## Features

- **Multi-format ingestion** — PDF, DOCX, TXT, CSV, MD, XLSX (up to 20MB each, 5 files at
  once), plus YouTube URLs via transcript extraction.
- **Grounded question answering** — semantic search in Qdrant, followed by a lexical/semantic
  reranking pass, followed by generation constrained to an anti-hedging prompt that refuses to
  answer beyond what was retrieved.
- **Document and video summarization** — a separate, wider "full context" retrieval mode for
  summary requests, distinct from the query-scoped search path used for regular questions.
- **Persistent, multi-turn memory** — per-session conversation history, spanning multiple
  documents attached to a single session, with older turns automatically compressed into an
  LLM-generated summary to keep long conversations token-efficient.
- **Source citations** — every RAG answer is tied back to the specific document, page, or
  section a claim came from.

---

## Architecture

```
Browser (templates/static/js/app_new.js)
        │
        ▼
FastAPI (app.py) ──── X-API-Key header ────▶ PostgreSQL (users, sessions, messages, attachments)
        │
        ├──▶ IngestionPipeline ──▶ local embedding model ──▶ Qdrant (per-document collection)
        │
        └──▶ QAPipeline
               ├──▶ MemoryManager       Postgres-backed: recent turns verbatim, older turns summarized
               ├──▶ Retriever           Qdrant search → lexical/semantic rerank → threshold filter
               └──▶ LangChain QA chain  prompt → Groq/Google LLM → sanitized, cited answer
```

---

## Design Decisions

### Why PostgreSQL

Chat sessions, messages, attachments, and user accounts are relational, multi-user data with
real query patterns behind them: *list this user's sessions by recency*, *prune messages older
than N days*, *look up a user by API key on every request*. Postgres provides this natively —
indexes, foreign keys with cascade delete, transactional writes — instead of hand-rolling it
against flat JSON files, which is what an earlier version of this project did
(`session_state_{id}.json`, `chat_history_{id}.json` per session). The schema
(`database/init.sql`) mirrors that earlier JSON structure table-for-table, so the change was a
storage-layer migration, not a data-model rewrite. It also gives straightforward multi-user
support and safe concurrent writes, neither of which flat files handle well once more than one
session is being written at a time.

### Why API-key authentication (not a login/password flow)

There is no username/password, session cookie, or login page. Every request carries an
`X-API-Key` header, which the backend hashes (SHA-256) and looks up against the `users` table
(`src/auth.py`). Keys are minted once, offline, with `python seed_user.py "Name"` — the
plaintext key is shown exactly once; only its hash is ever stored.

This is a deliberate trade-off for a project of this shape, typically self-hosted for a person
or a small team rather than a public multi-tenant SaaS product. A full login system — password
hashing, reset flows, session/cookie management, CSRF handling — is meaningful surface area to
secure correctly, for a benefit ("users set their own password") that matters less when
accounts are provisioned by whoever runs the server. An API key is simpler to reason about,
works identically for a browser client or a future CLI/API client, and revokes trivially
(`is_active = false`) without password-reset infrastructure. Public self-serve signup would be
a genuine, separate feature to add on top — not a sign the current approach is wrong for what
it serves today.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Backend + frontend | FastAPI, HTML templates, vanilla JS (`app_new.js`) |
| Vector store | Qdrant |
| Relational store | PostgreSQL |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally (CPU) |
| LLM | Groq (`llama-3.1-8b-instant`) or Google Gemini — config-switchable |
| Orchestration | LangChain (LCEL); LangGraph migration in progress on a separate branch |
| Reranking | Custom lexical/semantic-blended reranker (no external reranker model) |
| Auth | Hashed API keys, no login UI |

---

## Project Layout

```
app.py                      FastAPI app: routes, startup, request/response models
config/config.yaml          All tunables: embedding model, LLM provider/params, retriever
                             thresholds, chunking, upload limits, YouTube limits, Postgres conn
database/init.sql           Postgres schema (users, sessions, messages, attachments)
seed_user.py                 One-off script: create a user, issue an API key
src/
  auth.py                    X-API-Key → user_id dependency
  logger.py, exception.py    Shared logging + custom exception types
  components/
    embedder.py              Local embedding model wrapper
    vector_store.py          Qdrant client / collection management
    retriever.py             Search → rerank → threshold/margin filter → full-context mode
    memory_manager.py        Postgres-backed session/message/summary handling
    document_loader.py, text_splitter.py
  chains/qa_chain.py          Prompt, LLM factory, sanitizer, citation builder, the LCEL chain
  pipelines/
    ingestion_pipeline.py     File/YouTube → chunks → embeddings → Qdrant
    qa_pipeline.py             Orchestrates a /query request end to end
  utils/
    job_manager.py             Background job tracking for uploads
    rate_limiter.py            Proactive LLM rate-limit handling
    file_helper.py, youtube_helper.py
templates/                   Frontend HTML + static/js/app_new.js
tests/, evaluation/           Test suite and RAG evaluation scripts
```

---

## Setup

### Prerequisites

- Python 3.10
- A running **Qdrant** instance (default: `http://localhost:6333`)
  — `docker run -p 6333:6333 qdrant/qdrant`
- A running **PostgreSQL** instance
- A Groq API key (default LLM provider) and/or a Google API key

### 1. Environment variables

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

### 2. Python environment

```bash
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Database

```bash
createdb docmind
psql -d docmind -f database/init.sql
```

### 4. Create a user (API key)

```bash
python seed_user.py "Your Name"
```

Prints a `user_id` and a plaintext API key (`dk_live_...`) — copy it now, it is shown once.
This is the value any client sends as `X-API-Key`.

### 5. Configuration

`config/config.yaml` controls the LLM provider/model, retrieval thresholds
(`retriever.k`, `score_threshold`, `collection_margin`), chunking, upload limits, and YouTube
languages. Defaults are reasonable to start with.

### 6. Run

```bash
uvicorn app:app --reload
```

| URL | Purpose |
|---|---|
| `http://localhost:8000/` or `/app` | Chat UI |
| `http://localhost:8000/landing` | Landing page |
| `http://localhost:8000/health` | Health check (no auth required) |

All other routes require the `X-API-Key` header from step 4.

---

## API Reference

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

## Roadmap

`requirements.txt` already includes `langgraph` / `langgraph-checkpoint` / `langgraph-sdk`.
Orchestration is being migrated from a single LangChain LCEL chain to a LangGraph `StateGraph`
(retrieve → refine → generate, with token-by-token streaming to the frontend) on a separate
branch. Qdrant's collection structure and Postgres's schema are unaffected by that migration —
only `/query`'s internal orchestration and response streaming are changing.
