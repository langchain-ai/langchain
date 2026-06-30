# LangChain v1 Playground (web UI)

A small React + shadcn/ui frontend over a FastAPI backend that wraps the four
agents from the [`examples/`](../) suite — ReAct tools, structured extraction,
memory chat, and a todo planner. Each agent reuses the canonical `build_agent()`
from its example script, so the UI and the standalone scripts stay in sync.

```
examples/ui/
├── backend/      FastAPI app exposing the agents at /api/*
│   ├── agents.py   loads the four example agents once
│   ├── main.py     routes: /api/react /api/extract /api/chat /api/plan /api/health
│   └── run.sh      starts uvicorn via uv (ephemeral fastapi + langchain-openai)
└── frontend/     Vite + React + TypeScript + Tailwind + shadcn/ui
```

## Prerequisites

- The `langchain_v1` env is set up (`cd libs/langchain_v1 && uv sync`).
- Node 18+ (this was built against Node 22).
- An OpenAI API key.

## Run it (two terminals)

**Terminal 1 — backend** (serves on `http://localhost:8000`):

```bash
export OPENAI_API_KEY="sk-..."
examples/ui/backend/run.sh
```

`run.sh` launches uvicorn from `libs/langchain_v1` and pulls in `fastapi`,
`uvicorn`, and `langchain-openai` ephemerally with `uv run --with`, so no package
manifests change.

**Terminal 2 — frontend** (serves on `http://localhost:5173`):

```bash
cd examples/ui/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api/*` to the backend
on port 8000, so there is no CORS setup to do in development. The header badge shows
whether the backend is up and whether its OpenAI key is configured.

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET  | `/api/health`  | — | `{ ok, openai_key }` |
| POST | `/api/react`   | `{ question }` | `{ answer, tool_calls }` |
| POST | `/api/extract` | `{ text }` | `{ name, age, email }` |
| POST | `/api/chat`    | `{ thread_id, message }` | `{ reply }` |
| POST | `/api/plan`    | `{ goal }` | `{ todos, answer }` |

## Notes

- The backend builds all four agents at startup; the OpenAI key is only required
  when an endpoint is actually called (a missing key returns `503` with a clear
  message, surfaced in the UI).
- Chat memory is keyed by `thread_id`; the frontend generates one per browser
  session, so each tab is an independent conversation.
- `npm run build` runs `tsc` then `vite build` — a clean type-check plus a
  production bundle in `dist/`.
- The Vite proxy targets `http://localhost:8000` by default. To use a backend on
  another port, set `VITE_API_TARGET` and match the uvicorn `--port`, e.g.
  `VITE_API_TARGET=http://localhost:8001 npm run dev` with `... --port 8001`.
