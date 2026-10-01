# Router Documentation: `src/routers/pages.py`

## 1. Overview & Purpose
`src/routers/pages.py` renders HTML web pages for DocuVortex using template partial inclusion. It serves the landing page, the main chat workspace application, login redirects, and the favicon.

---

## 2. Endpoints & Route Definitions

### `GET /` & `GET /landing`
- **Response:** HTML template `templates/index.html` (Landing page showcasing architecture, capabilities, and call-to-actions).

### `GET /home` & `GET /app`
- **Response:** HTML template `templates/home.html` (Interactive DocuVortex workspace with sidebar, document attachment popover, chat typewriter view, and prompt composer).

### `GET /login`
- **Response:** HTTP 302 Redirect to `/app` (where the login and API key modals are seamlessly mounted).

### `GET /favicon.ico`
- **Response:** HTTP 204 No Content.

---

## 3. Connections & Component Mapping

- **Mounted In:** `app.py: app.include_router(pages_router)`.
- **Downstream Helper:** `src.utils.helpers: read_template` (parses `<!-- INCLUDE: partial_name -->` directives).
- **Template Directories:** `templates/`, `templates/partials/`.

---

## 4. AI & Developer Guidelines
- Frontend templates are composed modularly. If editing sidebar or header HTML, check `templates/partials/` first.
