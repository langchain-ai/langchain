# langchain-ai-audit-shelf

This package contains the LangChain integration for AI Audit Shelf.

## Installation

```bash
pip install -U langchain-ai-audit-shelf
```

## Usage

```python
from langchain_ai_audit_shelf import AIAuditCallbackHandler

handler = AIAuditCallbackHandler(
    api_url="http://localhost:8000",
    actor="my-agent"
)

# Pass the handler to your LLM or Chain
```
