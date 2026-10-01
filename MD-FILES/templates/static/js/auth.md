# Frontend Script Documentation: `templates/static/js/auth.js`

## 1. Overview & Purpose
`templates/static/js/auth.js` manages client-side authentication interactions across the login modal and dedicated login views. It fetches math captcha challenges from the backend, verifies user solutions, handles password recovery, and manages error displays.

---

## 2. Key Components & Functions

### `initCaptchaAuthFlow()`
Main lifecycle function mounted on `DOMContentLoaded`:
- **`loadCaptcha()`**: Fetches a fresh challenge from `/api/auth/captcha` and updates `#captchaQuestion` and `#captchaToken`.
- **`setLocalFallbackCaptcha()`**: Immediate offline math challenge generator ensuring the user never sees an empty challenge during slow network connections.
- **Form Submission**:
  - Gathers username, email, password, captcha answer, and captcha token.
  - Posts to `/api/auth/login`.
  - On success, redirects to `/app`.
  - On error, displays clear error text in `#authError`.

### Password Recovery
- Submits email to `/api/auth/forgot-password`.
- Displays retrieved password directly in a recovery alert box.

---

## 3. Connections & Component Mapping

- **Mounted In:** `templates/login.html` and the authentication modal in `templates/home.html`.
- **Backend Endpoints:**
  - `GET /api/auth/captcha`
  - `POST /api/auth/login`
  - `POST /api/auth/forgot-password`

---

## 4. AI & Developer Guidelines
- The captcha token input (`#captchaToken`) is a hidden input carrying the signed HMAC payload. Never wipe this input without calling `loadCaptcha()` to fetch a matching question.
