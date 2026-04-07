# langchain-ai-identity

Per-agent identity, policy enforcement, and tamper-proof audit trails for LangChain agents.

## Installation

```bash
pip install langchain-ai-identity
```

## Quick Start

```python
from langchain_ai_identity.middleware import AIIdentityGovernanceMiddleware

middleware = AIIdentityGovernanceMiddleware(
    agent_id="<your-agent-uuid>",
    api_key="aid_sk_...",
)

# Wrap a model call with governance enforcement
result = middleware.enforce_model_call(my_llm_call_fn, prompt="Hello")
```

## Callback Handler

```python
from langchain_ai_identity.callback import AIIdentityCallbackHandler

handler = AIIdentityCallbackHandler(
    api_key="aid_sk_...",
    agent_id="<your-agent-uuid>",
)
# Pass as a callback to any LangChain chain or agent
```

## Components

- **`AIIdentityCallbackHandler`** — Audit logging callback handler for LLM and tool events
- **`AIIdentityAsyncCallbackHandler`** — Async version of the callback handler
- **`AIIdentityGovernanceMiddleware`** — Middleware for per-agent policy enforcement and audit logging

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_IDENTITY_GATEWAY_URL` | `https://ai-identity-gateway.onrender.com` | Gateway endpoint |
| `AI_IDENTITY_API_URL` | `https://ai-identity-api.onrender.com` | Audit API endpoint |

## Links

- [AI Identity](https://www.ai-identity.co)
- [Documentation](https://www.ai-identity.co/docs)
- [PyPI](https://pypi.org/project/langchain-ai-identity/)
