# Module Documentation: `wipe_data.py`

## 1. Overview & Purpose
`wipe_data.py` is a comprehensive database reset and maintenance script. It flushes chat messages, attachments, sessions, and LangGraph checkpointer tables from PostgreSQL, and drops all vector collections from both remote Qdrant Docker and local embedded disk storage.

---

## 2. Key Components & Functions

### `wipe_postgres(include_users: bool = False) -> bool`
- Connects to Supabase PostgreSQL via `src.database.db:get_db_cursor()`.
- Truncates application tables with `CASCADE`: `messages`, `attachments`, `session_summaries`, `sessions`, `user_sessions`.
- Optionally truncates the `users` table if `--include-users` flag is passed.
- Truncates LangGraph checkpoint tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`.

### `wipe_qdrant() -> bool`
- Connects to Qdrant at `QDRANT_URL` (default `http://localhost:6333`).
- Iterates over all active vector collections and calls `client.delete_collection(col.name)`.
- Calls `_wipe_local_qdrant()` to clean up embedded disk storage (`data/qdrant_storage`).

### `_wipe_local_qdrant() -> bool`
- Drops embedded collections or deletes and recreates `data/qdrant_storage`.

---

## 3. Connections & Component Mapping

```mermaid
flowchart TD
    CLI["CLI (python wipe_data.py [--include-users])"] --> wipe["wipe_data.py"]
    wipe -->|TRUNCATE CASCADE| Postgres[("PostgreSQL Database (Supabase)")]
    wipe -->|delete_collection()| QdrantDocker[("Qdrant Remote Docker (Port 6333)")]
    wipe -->|rmtree / reset| LocalQdrant[("Local Qdrant Disk (data/qdrant_storage)")]
```

### Upstream Callers:
- Developers needing to clear test data, demo states, or corrupt sessions.

### Downstream Dependencies:
- `src.database.db`: PostgreSQL connection pool.
- `qdrant_client`: Vector database SDK.

---

## 4. AI & Developer Guidelines
- **Data Loss Warning:** Running this script is irreversible. It completely wipes chat history and vector embeddings.
- By default, user accounts are preserved so API keys remain valid. Passing `--include-users` requires re-running `seed_user.py`.
