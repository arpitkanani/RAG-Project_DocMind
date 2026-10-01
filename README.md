# 🌪️ DocuVortex: Intelligent Multi-Modal RAG & Agentic Document Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C.svg)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg?logo=supabase&logoColor=white)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DocuVortex** is an enterprise-grade, agentic Retrieval-Augmented Generation (RAG) platform designed to converse with diverse document collections and YouTube video transcripts with zero hallucination, verifiable citations, persistent chat memory, and lightning-fast semantic retrieval.

Built on **LangGraph state machines**, **Qdrant vector databases**, and **Supabase PostgreSQL**, DocuVortex transforms raw unstructured data into an interactive, multi-turn AI workspace.

---

## 🌟 Key Features

- 📄 **Multi-Format Document Ingestion:** Native parsing and chunking for PDF, DOCX, TXT, CSV, JSON, and Markdown documents.
- 🎥 **YouTube Video Intelligence:** Ingests YouTube URLs, retrieves accurate transcripts with timestamps, and enables conversational video analysis.
- 🧠 **Agentic LangGraph Workflow:** Query rewriting, adaptive retrieval, context compression, hallucination verification, and strict citation grounding.
- 🔑 **Frictionless API-Key Authentication:** Secure API-key-first authentication generated via CLI (`seed_user.py`), cached in localStorage, and transmitted via `X-API-Key` headers. Includes a built-in guest mode fallback for immediate exploration.
- 💾 **Stateful Conversational Memory:** Multi-turn session persistence backed by Supabase PostgreSQL `AsyncPostgresSaver` checkpointer, sliding context windows, and automatic background session summarization.
- ⚡ **High-Speed Vector Search:** Powered by Qdrant with dense embeddings (`BAAI/bge-small-en-v1.5` or OpenAI embeddings).
- 🎨 **Modern Single-Page UI:** Clean, responsive dark/light interface with drag-and-drop file uploads, real-time citation panels, and chat history switching.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client["Web Frontend / API Consumers"]
        UI["Single Page App (HTML5 / Vanilla JS)"]
        CLI["Script / External API Client"]
    end

    subgraph API["FastAPI Backend (app.py)"]
        AuthMiddleware["Auth & API Key Resolver (src/auth.py)"]
        RouterUpload["Upload Router (/upload)"]
        RouterYT["YouTube Router (/youtube)"]
        RouterQuery["Query Router (/query)"]
        RouterSessions["Sessions Router (/sessions)"]
        RouterAuth["Auth Router (/api/auth)"]
    end

    subgraph Pipelines["Processing & StateGraph"]
        DocIngest["Ingestion Pipeline (PyMuPDF / Chunking)"]
        Embedder["Embedder (HuggingFace / OpenAI)"]
        StateGraph["LangGraph Multi-Agent RAG Graph"]
    end

    subgraph Storage["Persistence & Vectors"]
        Qdrant[("Qdrant Vector Database")]
        Postgres[("Supabase PostgreSQL (Sessions, Messages, Users)")]
    end

    UI -->|X-API-Key / REST| API
    CLI -->|X-API-Key / REST| API
    API --> AuthMiddleware
    AuthMiddleware --> Postgres

    RouterUpload --> DocIngest --> Embedder --> Qdrant
    RouterYT --> Embedder --> Qdrant
    RouterQuery --> StateGraph
    StateGraph <--> Qdrant
    StateGraph <--> Postgres
    RouterSessions <--> Postgres
```

---

## 📂 Project Structure

```text
├── app.py                      # FastAPI application entrypoint & lifespan lifecycle
├── seed_user.py                # CLI script to generate users & dk_live_... API keys
├── check_db.py                 # Diagnostic database connection & statistics tool
├── wipe_data.py                # Safe development data cleanup script
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment configuration
│
├── config/
│   └── config.yaml             # Model, chunking, and database defaults
│
├── database/
│   └── init.sql                # PostgreSQL relational schema for Supabase
│
├── src/
│   ├── auth.py                 # API key hash verification & JWT utilities
│   ├── logger.py               # Centralized rotating file & console logger
│   ├── exception.py            # Custom exception wrapper with traceback details
│   │
│   ├── components/
│   │   ├── embedder.py         # HuggingFace & OpenAI embedding provider
│   │   ├── ingestion.py        # Document text extraction & semantic chunking
│   │   ├── memory_manager.py   # Multi-turn history, attachments, & summaries
│   │   └── vector_store.py     # Qdrant client connection and collection manager
│   │
│   ├── database/
│   │   └── db.py               # psycopg2 threaded connection pool manager
│   │
│   ├── graph/
│   │   ├── builder.py          # LangGraph StateGraph definition & compiled graph
│   │   ├── nodes.py            # Graph execution nodes (retrieve, generate, rewrite)
│   │   └── state.py            # RAGState TypedDict state schema
│   │
│   ├── routers/
│   │   ├── auth.py             # /api/auth routes (verify-key, session-status, logout)
│   │   ├── collections.py      # Vector collection management endpoints
│   │   ├── health.py           # System liveness & readiness checks
│   │   ├── pages.py            # HTML template rendering (/app, /landing)
│   │   ├── query.py            # LangGraph RAG conversational query endpoint
│   │   ├── sessions.py         # Chat session listing, history, & deletion
│   │   ├── upload.py           # Document upload & ingestion endpoint
│   │   └── youtube.py          # YouTube video transcription & indexing
│   │
│   └── utils/
│       └── helpers.py          # Template readers & sanitization helpers
│
├── templates/
│   ├── home.html               # Main workspace SPA interface
│   ├── index.html              # Landing page
│   └── static/
│       ├── css/                # Design stylesheets
│       └── js/                 # Vanilla JS app logic (app_new.js)
│
└── MD-FILES/                   # Complete architectural documentation per module
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+**
- **Qdrant Vector Database** (Running locally on `http://localhost:6333` or Qdrant Cloud)
- **Supabase PostgreSQL** (or standard PostgreSQL 14+)

### 2. Clone & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/arpitkanani/RAG-Project_DocMind.git
cd "RAG-Project_DocMind"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Database Settings (Supabase)
# Port 6543 for pooled app queries, Port 5432 for LangGraph prepared statements
DATABASE_URL=postgresql://postgres.xxx:yourpassword@aws-0-region.pooler.supabase.com:6543/postgres
LANGGRAPH_DATABASE_URL=postgresql://postgres.xxx:yourpassword@aws-0-region.supabase.com:5432/postgres

# Qdrant Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LLM Providers (Configure at least one)
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...

# Application Security
SECRET_KEY=generate_a_secure_random_32_character_string_here
```

### 4. Seed an Admin User & Generate API Key
Run the user seeding CLI to create your user profile and generate a live API key:
```bash
python seed_user.py "my_username"
```
Output:
```text
=================================================================
DocuVortex — Seeding User & Generating API Key
=================================================================
✅ SUCCESS! User successfully created / updated.
User Name : my_username
User ID   : e4d99fc3-b788-466f-bf3b-240fae41cbdb
Email     : my_username@docuvortex.local

YOUR API KEY (copy and paste this into the browser popup):
    dk_live_9f81a7b63c4d2e1084f7a213e8d9...
=================================================================
```

### 5. Start the Application
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open your browser and navigate to:
- **Application Workspace:** `http://localhost:8000/app`
- **Interactive OpenAPI Docs:** `http://localhost:8000/docs`

When the app opens, paste the generated `dk_live_...` API key into the popup dialog (or click **"Continue as Guest"** for quick exploration).

---

## 🔐 Authentication Model

DocuVortex uses a simple and secure **API-key-first authentication architecture**:
1. **Key Generation:** Keys are generated via `seed_user.py` using cryptographically secure tokens formatted as `dk_live_<48-hex-chars>`.
2. **Database Storage:** The plain key is shown to the user once upon creation. Only its **SHA-256 hash** is stored in the `users.api_key_hash` database column.
3. **API Requests:** Clients pass the key in the `X-API-Key` HTTP header:
   ```bash
   curl -X POST "http://localhost:8000/query" \
     -H "X-API-Key: dk_live_9f81a7b63c4d..." \
     -H "Content-Type: application/json" \
     -d '{"question": "Summarize the uploaded document", "session_id": "test-session"}'
   ```
4. **Session Recovery:** On initial key entry, the backend automatically looks up the user's previous chat `session_id`, instantly restoring their conversation history.
5. **Guest Mode:** If no key is configured, users can click "Continue as Guest" to explore with a default guest UUID (`00000000-0000-0000-0000-000000000001`).

---

## 📡 API Reference Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/app` | Main single-page interactive chat interface |
| `POST` | `/api/auth/verify-key` | Validate API key and issue 30-day session token |
| `GET` | `/api/auth/session-status` | Check authentication status & retrieve active session ID |
| `POST` | `/api/auth/logout` | Revoke session and delete session cookie |
| `POST` | `/upload` | Ingest and embed PDF, DOCX, TXT, CSV, JSON files |
| `POST` | `/youtube` | Ingest, transcribe, and index YouTube video content |
| `POST` | `/query` | Execute LangGraph conversational RAG query |
| `GET` | `/sessions` | List all chat sessions for the authenticated user |
| `GET` | `/sessions/{id}` | Retrieve full message history and citations for a session |
| `DELETE`| `/sessions/{id}` | Permanently delete a chat session and associated metadata |
| `GET` | `/health` | Check PostgreSQL pool and vector store health |

---

## 📚 Complete Technical Documentation

Comprehensive documentation for every single module, database schema, and router is available in the [`MD-FILES/`](MD-FILES/) directory:

- 📖 [`MD-FILES/README.md`](MD-FILES/README.md) - Documentation Directory Index
- 🔐 [`MD-FILES/src/auth.md`](MD-FILES/src/auth.md) - Security & Authentication Architecture
- 🌐 [`MD-FILES/src/routers/auth.md`](MD-FILES/src/routers/auth.md) - Auth REST API Reference
- 🗄️ [`MD-FILES/database/init.md`](MD-FILES/database/init.md) - Relational Schema & ERD
- 🔌 [`MD-FILES/src/database/db.md`](MD-FILES/src/database/db.md) - PostgreSQL Connection Pooling
- 🧠 [`MD-FILES/src/graph/builder.md`](MD-FILES/src/graph/builder.md) - LangGraph Workflow Engine

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
