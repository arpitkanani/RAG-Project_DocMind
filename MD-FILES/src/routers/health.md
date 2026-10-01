# Router Documentation: `src/routers/health.py`

## 1. Overview & Purpose
`src/routers/health.py` provides a lightweight liveness probe endpoint for DocuVortex.

---

## 2. Endpoints & Route Definitions

### `GET /health`
- **Purpose:** System uptime and connectivity verification for Docker healthchecks, Kubernetes probes, and reverse proxies.
- **Handler:** Returns `{"status": "ok", "service": "DocuVortex API"}` with HTTP 200.

---

## 3. Connections & Component Mapping

- **Mounted In:** `app.py: app.include_router(health_router)`.
- **Used By:** `docker-compose.yml` healthcheck, `endpoint_regression_check.py`.
