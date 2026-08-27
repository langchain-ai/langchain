# `langchain.mcp` examples

Runnable, self-contained scripts. Each starts whatever MCP server it needs, so
there is nothing to launch separately.

```bash
uv sync --extra mcp --extra anthropic
export ANTHROPIC_API_KEY=...          # needed by the examples that run an agent
uv run examples/mcp/transports.py
```

| Example | Shows | Needs a model |
|---|---|:-:|
| `transports.py` | one adapter over in-memory, stdio, and HTTP | |
| `multi_server.py` | several servers behind one adapter, tools prefixed per server | ✅ |
| `tool_errors.py` | a failing tool reaching the model so it can retry | ✅ |
| `elicitation.py` | a server asking a human mid-call, via `interrupt()` | ✅ |
| `auth_bearer.py` | a server behind a static bearer token | |
| `auth_oauth.py` | a full OAuth 2.1 flow with dynamic client registration | |

`auth_oauth.py` opens a browser tab. The demo authorization server auto-approves,
so it redirects straight back.

`_servers.py` holds the small MCP servers the examples share, and
`_stdio_server.py` is the one script `transports.py` launches over stdio.
Neither is part of the API being demonstrated.
