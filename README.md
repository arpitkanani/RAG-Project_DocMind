# 🌪️ DocuVortex: Intelligent Multi-Modal RAG & Agentic Document Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C.svg)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg?logo=supabase&logoColor=white)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DocuVortex** is an enterprise-grade, agentic Retrieval-Augmented Generation (RAG) platform designed to converse with diverse document collections and YouTube video transcripts with zero hallucination, verifiable citations, persistent chat memory, and multi-tier model fallbacks.

Built on **LangGraph state machines**, **Qdrant vector databases**, and **Supabase PostgreSQL**, DocuVortex transforms raw unstructured data into an interactive, multi-turn AI workspace.

---

## 🌟 Key Features

### 1. Document & Media RAG Workspace (`/app`)
- 📄 **Multi-Format Document Ingestion:** Native parsing and semantic chunking for PDF, DOCX, TXT, CSV, XLSX, and Markdown documents.
- 🎥 **YouTube Video Intelligence:** Ingests YouTube URLs, retrieves accurate transcripts with timestamps, and enables conversational video analysis.
- 🧠 **Agentic LangGraph Workflow:** Query rewriting, adaptive retrieval, context compression, hallucination verification, and strict citation grounding.
- 🛡️ **3-Tier LLM Generation Fallback:** Automatic zero-downtime generation routing:
  $$\text{All Groq Models} \longrightarrow \text{All Gemini Models} \longrightarrow \text{Azure OpenAI } \texttt{gpt-5.4-nano}$$
- ⚡ **Multi-Provider Embeddings & Strict Consistency:**
  - **Primary:** Azure OpenAI `text-embedding-3-small` (1536 dims) with fail-fast resilience (`ResilientAzureEmbeddings`).
  - **Fallback:** Local `BAAI/bge-small-en-v1.5` (384 dims) loaded lazily only when selected by the user.
  - **Per-Collection Consistency:** Collections are audited against active vector dimensions; incompatible documents display a `"Needs re-index"` warning badge.
- ⚙️ **Workspace Settings UI:** In-app settings modal allowing instant switching between Azure OpenAI and local HuggingFace embeddings.
- 💾 **Stateful Conversational Memory:** Multi-turn session persistence backed by Supabase PostgreSQL `AsyncPostgresSaver` checkpointer, 10-turn (20 messages) verbatim context window, and batched background session summarization to minimize redundant LLM calls.

### 2. Custom Business Bot Platform & Architect Wizard (`/builder`, `/admin-dashboard`, `/bot/{bot_id}`)
- 🎙️ **Live Master Architect WebSocket Interview:** Interactive interview conducted by Azure OpenAI probing business domain across 10 critical operational vectors with real-time badge tracking.
- 🛡️ **Dynamic Prompt Synthesis & Safety Shields:** Automatically compiles enterprise system prompts with verified knowledge boundaries, contact info, and anti-leak prompt-injection shields. Includes advisory guidance for two-step LLM verification before deployment.
- 🌐 **Single-Page Website Scraping & Multimodal Brochure Ingestion:** Up to 5 website URLs and 5 PDF brochures (&le; 30 MB each, dynamically configurable via `config/config.yaml`) ingested into isolated Qdrant vector collections (`biz_{bot_id}`) with multimodal Groq Vision OCR (`llama-3.2-11b-vision-preview`) for brochure images, scans, and tables. Features browser export tips (`Ctrl + P` to PDF).
- 💬 **Embeddable Customer Chat (`/bot/{bot_id}`):** High-concurrency chat with typewriter streaming, Warm Ivory glassmorphic header, and "Back to Overview" navigation pill.
- 📋 **Automatic Lead Capture & Privacy Shield:** Automatically extracts customer contact inquiries (name, email, phone) into PostgreSQL `bot_leads` without echoing raw credentials back to users.

### 3. Automated Browser Testing & Security
- 🧪 **117 Automated Playwright Tests:** End-to-end browser tests verifying responsive layouts across **19 viewport dimensions** (`320x568` to `1920x1080`) with zero horizontal overflow.
- 🔒 **Anti-Leak Security Suite:** Scans public endpoints against credential/connection string leaks, rejects unauthenticated access (401/303), and enforces multi-tenant role isolation (403 Forbidden).

---

## 🖥️ The Five Unified Workspaces

| Workspace | Route | Target User | Key Capabilities |
| :--- | :--- | :--- | :--- |
| **1. Landing Page** | `/` | Guests, Users, Admins | Assistant Directory showcase, RAG vs Business Bot cards, unified Auth modal, role-based navigation. |
| **2. Document RAG** | `/app` | Researchers, Analysts | Multi-document upload, YouTube transcript indexing, source filter popover, LangGraph citations. |
| **3. Bot Architect** | `/builder` | Business Admins | 4-step wizard: website scraping, brochure upload, Master Architect live interview, prompt synthesis. |
| **4. Admin Dashboard**| `/admin-dashboard` | Business Admins | My Bots directory, bot launcher, delete confirmation, Indian date formatting, inquiries & leads table. |
| **5. Business Chat** | `/bot/{bot_id}` | End Customers | Dedicated chat with initial greeting chips, brochure RAG grounding, and appointment lead capture. |

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
        RouterSettings["Settings Router (/api/settings)"]
        RouterSessions["Sessions Router (/sessions)"]
        RouterAuth["Auth Router (/api/auth)"]
    end

    subgraph Pipelines["Processing & StateGraph"]
        DocIngest["Ingestion Pipeline (PyMuPDF / Chunking)"]
        Embedder["Embedder (Azure 1536-dim / BGE 384-dim)"]
        StateGraph["LangGraph Multi-Agent RAG Graph"]
        FallbackLLMs["LLM Fallback (Groq -> Gemini -> Azure gpt-5.4-nano)"]
    end

    subgraph Storage["Persistence & Vectors"]
        Qdrant[("Qdrant Vector Database")]
        Postgres[("Supabase PostgreSQL (Users, Settings, Sessions, Messages)")]
    end

    UI -->|X-API-Key / REST| API
    CLI -->|X-API-Key / REST| API
    API --> AuthMiddleware
    AuthMiddleware --> Postgres

    RouterUpload --> DocIngest --> Embedder --> Qdrant
    RouterYT --> Embedder --> Qdrant
    RouterQuery --> StateGraph
    StateGraph --> FallbackLLMs
    StateGraph <--> Qdrant
    StateGraph <--> Postgres
    RouterSessions <--> Postgres
    RouterSettings <--> Postgres
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
│   ├── chains/
│   │   └── qa_chain.py         # Grounded QA chain & _build_llm() fallback hierarchy
│   │
│   ├── components/
│   │   ├── embedder.py         # Multi-provider embedder (Azure & BGE lazy singleton)
│   │   ├── ingestion.py        # Document text extraction & semantic chunking
│   │   ├── memory_manager.py   # Multi-turn history, attachments, & dimension checks
│   │   └── vector_store.py     # Qdrant client connection & dimension guard
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
│   │   ├── auth.py             # /api/auth routes
│   │   ├── settings.py         # /api/settings routes (embedding provider toggle)
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
│   ├── home.html               # Main workspace SPA interface & Settings modal
│   ├── index.html              # Landing page
│   └── static/
│       ├── css/                # Design stylesheets
│       └── js/                 # Vanilla JS app logic (app_new.js)
```

---

## 📡 API Reference Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/app` | Main single-page interactive chat interface |
| `GET` | `/api/settings/embedding` | Get active provider & available model metadata |
| `POST` | `/api/settings/embedding` | Switch user's active embedding provider (`azure` or `bge`) |
| `POST` | `/api/auth/create-key` | In-browser API key generation and identity rotation |
| `POST` | `/api/auth/verify-key` | Validate API key and issue 30-day session token |
| `GET` | `/api/auth/session-status` | Check authentication status & retrieve active session ID |
| `POST` | `/api/auth/logout` | Revoke session and delete session cookie |
| `POST` | `/upload` | Ingest and embed PDF, DOCX, TXT, CSV, JSON files |
| `POST` | `/youtube` | Ingest, transcribe, and index YouTube video content |
| `POST` | `/query` | Execute LangGraph conversational RAG query (SSE stream) |
| `GET` | `/sessions` | List all chat sessions for the authenticated user |
| `GET` | `/sessions/{id}` | Retrieve full message history, citations, and attachment audit |
| `DELETE`| `/sessions/{id}` | Permanently delete a chat session and associated metadata |
| `GET` | `/health` | Check PostgreSQL pool and vector store health |
| `GET` | `/builder` | Bot Architect Wizard page (guarded: requires Admin) |
| `GET` | `/admin-dashboard` | Admin control center and leads table (guarded: requires Admin) |
| `GET` | `/bot/{bot_id}` | Embeddable public customer bot chat interface |
| `GET` | `/api/business-bot/public/list` | List active public bots for the Assistant Directory |
| `GET` | `/api/business-bot/admin/bots` | List bots belonging to authenticated admin |
| `DELETE`| `/api/business-bot/admin/bots/{id}` | Permanently delete bot, Qdrant collection, and leads |
| `GET` | `/api/business-bot/admin/leads/{id}` | Retrieve customer inquiry leads captured for bot |
| `POST` | `/api/business-bot/scrape-url` | Scrape up to 5 website URLs without recursive crawling |
| `POST` | `/api/business-bot/upload-brochures` | Upload up to 3 PDF brochures (max 4MB each) |
| `WS` | `/ws/business-bot/interview/{id}` | Real-time WebSocket Master Architect interview stream |
| `WS` | `/ws/business-bot/chat/{id}` | Real-time WebSocket customer chat stream with tool RAG |

---

## 🧪 Automated Testing Suite (Playwright)

DocuVortex includes a 117-test Playwright suite testing real Chromium rendering across 19 viewports:

```bash
# Run all 117 tests headlessly
npx playwright test

# Run tests with visible browser window
npx playwright test --headed

# Open the Playwright interactive UI runner
npx playwright test --ui

# View HTML report
npx playwright show-report
```

---

## 🚀 Azure Container Apps Deployment

Deploy the containerized app with WebSocket sticky sessions and pre-baked embeddings:

```bash
# Make deployment script executable and deploy to Azure
chmod +x scripts/deploy_azure.sh
./scripts/deploy_azure.sh
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
