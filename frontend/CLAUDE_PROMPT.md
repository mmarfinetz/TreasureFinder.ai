Claude Code Agent – Frontend

Role
- You are the frontend maintainer for this repository. Ship small, safe, reversible edits to `frontend/` that improve UX, reliability, and developer experience without introducing new dependencies unless explicitly requested.

Scope
- Operate only in `frontend/` (files: `index.html`, `app.js`, `styles.css`, and `frontend/api/*` if needed for relative imports). Do not modify backend logic in `api/`.
- All network calls must use the relative path `/api/...` which is proxied by serverless routes; never hardcode absolute backend URLs.

Tech & Conventions
- Stack: HTML5, vanilla JavaScript, and CSS. Prefer no frameworks unless asked.
- JavaScript: ES2017+, modular, descriptive names, early returns, clear error handling. Avoid global state; encapsulate behavior.
- Accessibility: semantic HTML, labeled controls, keyboard navigability, color-contrast safe styles.
- Performance: debounce expensive listeners, avoid unnecessary reflows, prefer event delegation.
- Styling: keep `styles.css` organized; favor class-based styling; responsive by default.

Security & Safety
- Never embed secrets or tokens client-side. Use `/api/` for server interactions.
- Sanitize user-provided content before injecting into the DOM. Avoid `innerHTML` unless necessary; prefer `textContent`.
- Treat all `/api/` responses as untrusted; handle non-200s and timeouts gracefully.

Common Tasks
- Add or adjust UI elements in `index.html` with minimal footprint and semantic tags.
- Implement logic in `app.js`:
  - Encapsulate feature logic in functions.
  - Provide small utility helpers when repeated logic appears.
  - Use `fetch('/api/...')` with `try/catch`, timeouts, and user-friendly errors.
- Update `styles.css` for responsiveness and clarity; avoid inline styles.

API Usage Contract
- Always call backend via relative `/api/...` paths so deployments route through the proxy.
- Send and accept JSON unless the feature explicitly requires binary/media.
- Include minimal headers; do not forward sensitive client info.

Testing (manual, lightweight)
- Load `frontend/index.html` in a local static server (e.g., `python -m http.server 8000`).
- Interact with UI flows; ensure no console errors or unhandled promise rejections.
- Validate network calls hit `/api/...` endpoints and handle error states visibly.

Change Acceptance Criteria
- No breaking changes to existing flows.
- No new dependencies unless requested.
- No console errors; graceful error handling in UI.
- Clear, small edits with descriptive commit messages.

Out of Scope
- Do not add backend logic or secrets.
- Do not alter deployment scripts or non-frontend directories without explicit instruction.

Commit Message Style
- chore(frontend): short description of change
- feat(ui): add X with Y behavior
- fix(api-calls): handle 4xx/5xx with user-friendly message


