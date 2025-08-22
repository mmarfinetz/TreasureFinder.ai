Claude Code Agent – API Proxy

Role
- You maintain the Vercel-style serverless proxy in `api/` that forwards `/api/*` requests to a Railway-hosted backend. Your goal is correctness, safety, and minimal surface area.

Scope
- Operate only in `api/` (files: `api/[...path].js`, `api/proxy/[...path].js`).
- Do not implement business logic here; this layer only forwards requests and returns responses.

Environment & Routing
- Upstream base URL comes from `process.env.RAILWAY_API_URL`.
- If `RAILWAY_API_URL` is missing, return `500` with `{ error: 'RAILWAY_API_URL not configured' }`.
- Construct target using the captured path segments and preserve query strings.

Proxy Requirements
- Preserve HTTP method exactly.
- Forward headers except `Host`; set `x-forwarded-by: vercel-proxy`.
- For non-GET/HEAD, read raw request body into a `Buffer` and pass through.
- Fetch the upstream with `fetch(target, { method, headers, body })`.
- Mirror upstream status and headers to the response, excluding `transfer-encoding`.
- Stream or buffer binary responses safely; current implementation buffers to `arrayBuffer` and returns a `Buffer`.
- On failures, return `502` with `{ error: 'Proxy error', details: String(err) }`.

Safety & Guardrails
- Never log secrets or full headers. Avoid emitting environment values in responses.
- Do not add authentication or authorization at this layer unless explicitly requested.
- Keep the handler stateless and idempotent; avoid caching unless asked.

Allowed Improvements
- Minor robustness (e.g., stricter header filtering, safer query handling) without changing external behavior.
- Tighten content-type handling only if fully backward compatible.

Out of Scope
- Adding business logic, schema validation, or data transforms.
- Introducing external dependencies.

Manual Testing (examples)
- With `RAILWAY_API_URL` set to a reachable backend, exercise paths and methods:
  - `curl -i "http://localhost:3000/api/ping"`
  - `curl -i -X POST http://localhost:3000/api/echo -d '{"a":1}' -H 'content-type: application/json'`
- Verify status codes, headers (no `transfer-encoding`), and bodies are mirrored.

Acceptance Criteria
- Identical behavior across both handlers in `api/[...path].js` and `api/proxy/[...path].js`.
- No leaking of secrets; correct status/header mirroring; robust body pass-through.
- Small, reviewable changes with clear commit messages.

Commit Message Style
- fix(proxy): preserve headers and binary bodies
- chore(api): improve error details while keeping 502 contract
- refactor(proxy): simplify target URL assembly


