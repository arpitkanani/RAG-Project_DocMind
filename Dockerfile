# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – builder: install Python deps into a venv so the final image is lean
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# OS-level build deps (needed by psycopg2-binary, torch wheels, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv so we can copy it cleanly to the final stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip/wheel first for faster installs
RUN pip install --upgrade pip wheel

# Copy only the requirements file first (better layer caching)
COPY requirements.txt .

# Install runtime requirements
RUN pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – final runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime OS libraries only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN addgroup --system docmind && adduser --system --ingroup docmind docmind

# Copy application source code
COPY --chown=docmind:docmind . .

# Set Hugging Face cache dir to a known path
ENV HF_HOME=/home/docmind/.cache/huggingface

# Ensure upload / log / cache directories exist and are writable by the app user
RUN mkdir -p data/uploads logs /home/docmind/.cache/huggingface \
    && chown -R docmind:docmind data logs /home/docmind

# Pre-download the embedding model at build time into the image cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    && chown -R docmind:docmind /home/docmind/.cache

USER docmind

# Uvicorn listens on this port; Docker Compose / Kubernetes exposes it
EXPOSE 8000

# Health-check so Docker knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
