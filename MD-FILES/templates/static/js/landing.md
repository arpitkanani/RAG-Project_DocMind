# Frontend Script Documentation: `templates/static/js/landing.js`

## 1. Overview & Purpose
`templates/static/js/landing.js` is the lightweight client script for the DocuVortex landing page (`templates/index.html`).

---

## 2. Key Components & Functions

- Binds click listeners to all elements matching `.trigger-login-modal`.
- Smoothly redirects visitors to `/app` where the full interactive workspace and authentication popups are mounted.

---

## 3. Connections & Component Mapping

- **Mounted In:** `templates/index.html`.
- **Target URL:** `/app` (`src.routers.pages: chat_app`).
