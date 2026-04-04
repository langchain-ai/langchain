# Headless tool + LangGraph dev server (E2E)

This example runs a **local LangGraph API** (`langgraph dev`) with a graph built via
[`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) and a **headless
tool** from the branch (`langchain.tools.tool` with `name`, `description`, and
`args_schema` only). The tool calls LangGraph `interrupt()` so you can handle execution in
a browser or other client and **resume** with the result.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- An `OPENAI_API_KEY` in `.env` (the demo model defaults to `openai:gpt-4o-mini`)

## 1. Configure environment

```bash
cd examples/headless-langgraph-e2e
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## 2. Install dependencies

```bash
uv sync --group dev
```

## 3. Start the LangGraph dev server

Default API URL is `http://127.0.0.1:2024` (CORS allows `*` by default).

```bash
uv run langgraph dev --config langgraph.json --no-browser
```

The dev server listens on **port 2024** by default (see the banner for the exact API URL).

### Port 2024 already in use

Another process (often a previous `langgraph dev`) is bound to 2024. Either stop it or use a free port:

```bash
# See what is using 2024 (macOS / Linux)
lsof -i :2024
```

Then quit that process, or start on another port:

```bash
uv run langgraph dev --config langgraph.json --no-browser --port 2025
```

Set the **LangGraph API base URL** in the frontend to match (for example `http://127.0.0.1:2025`).

## 4. Run the React frontend (Vite + `useStream`)

The UI mirrors the headless-tools pattern from `ui-playground` (`tool({ name, description, schema })` +
`.implement(...)` passed to `useStream({ tools: [...] })`).

In another shell:

```bash
cd examples/headless-langgraph-e2e/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:8765`. The **LangGraph API base URL** field defaults to
`http://127.0.0.1:2024`; change it if you used `--port` with a different value.

Send a message that asks to open a URL or for your location; the model should call the
matching headless tool, the client runs the implementation in `src/tools.ts`, and streaming
continues after the tool result is applied.

## Notes

- **Checkpointer:** For a plain Python script you normally pass `checkpointer=InMemorySaver()`
  to `create_agent` so `interrupt()` / headless tools persist. The **LangGraph dev server**
  injects its own persistence and **rejects** a custom checkpointer on the graph, so this
  example omits it.
- **Studio:** The server banner prints a link to open **LangSmith Studio** against your
  local API (useful for debugging the same graph).

## Graph id

The graph is registered as **`agent`** in `langgraph.json` (this is the assistant / graph
id passed to the SDK).
