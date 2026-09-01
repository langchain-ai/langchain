# `langchain.mcp` examples

Runnable, self-contained scripts. Each starts whatever MCP server it needs, so
there is nothing to launch separately.

```bash
uv sync --extra mcp --extra anthropic
export ANTHROPIC_API_KEY=...          # needed by the examples that run an agent
uv run examples/mcp/transports.py
```

| Example | Shows | Model | Network |
|---|---|:-:|:-:|
| `transports.py` | one adapter over in-memory, stdio, and HTTP | | |
| `remote_server.py` | pointing the adapter at a public MCP server | ✅ | ✅ |
| `multi_server.py` | several servers behind one adapter, tools prefixed per server | ✅ | |
| `graph_factory.py` | one long-lived adapter shared by every run of a `langgraph dev` graph | | |
| `protocol_eras.py` | one agent holding tools from both MCP protocol eras | ✅ | |
| `tool_errors.py` | a failing tool reaching the model so it can retry | ✅ | |
| `elicitation.py` | a server asking a human mid-call, via `interrupt()` | ✅ | |
| `destructive_interrupt.py` | gating destructive tools behind approval, from tool metadata | ✅ | |
| `auth_bearer.py` | a server behind a static bearer token | | |
| `auth_oauth.py` | a full OAuth 2.1 flow with dynamic client registration | | |

`remote_server.py` calls DeepWiki, a public MCP server, so it needs internet
access. `auth_oauth.py` opens a browser tab; the demo authorization server
auto-approves, so it redirects straight back.

`_servers.py` holds the small MCP servers the examples share, and
`_stdio_server.py` is the entry point launched as a subprocess over stdio.
Neither is part of the API being demonstrated.
