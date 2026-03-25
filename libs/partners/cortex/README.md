# langchain-cortex

LangChain integration for [Cortex](https://github.com/gambletan/cortex) — a
local-first persistent memory engine for AI agents.

## What is Cortex?

Cortex is a Rust-powered memory engine with:

- **4-tier memory**: Working → Episodic → Semantic → Procedural
- **HNSW vector search**: Sub-millisecond at 100K+ memories
- **Bayesian beliefs**: Self-correcting with evidence
- **People graph**: Cross-channel identity resolution
- **Zero cloud**: 100% local, your data stays on your device

## Installation

```bash
pip install langchain-cortex cortex-ai-memory
```

## Quick Start

### CortexChatMessageHistory

Drop-in replacement for any `BaseChatMessageHistory` backend:

```python
from langchain_cortex import CortexChatMessageHistory

history = CortexChatMessageHistory(
    db_path="~/.cortex/chat.db",
    session_id="my-session",
)

history.add_user_message("What is the capital of France?")
history.add_ai_message("Paris.")

for msg in history.messages:
    print(msg)

history.clear()
```

### CortexMemory

Drop-in replacement for `ConversationBufferMemory` with persistent storage:

```python
from langchain_cortex import CortexMemory

memory = CortexMemory(
    db_path="~/.cortex/agent.db",
    session_id="agent-session-1",
    memory_key="history",
)

memory.save_context(
    {"input": "My name is Alvin."},
    {"output": "Nice to meet you, Alvin!"},
)

vars_ = memory.load_memory_variables({})
# {'history': [HumanMessage(content='My name is Alvin.'), AIMessage(...)]}
```

### With RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cortex import CortexChatMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: CortexChatMessageHistory(
        db_path="~/.cortex/chat.db",
        session_id=session_id,
    ),
    input_messages_key="input",
    history_messages_key="history",
)

response = chain_with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user-42"}},
)
```

### Structured knowledge alongside chat

```python
from langchain_cortex import CortexMemory

memory = CortexMemory(db_path="~/.cortex/memory.db", session_id="demo")

# Store semantic facts
memory.add_fact("User", "works_at", "Acme Corp", confidence=0.95)

# Store preferences
memory.add_preference("timezone", "Asia/Shanghai", confidence=0.9)

# Multi-tier context summary (better than raw chat buffer for system prompts)
context = memory.get_context(max_tokens=2000)
```

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` | — | Path to Cortex SQLite file (`~` expanded) |
| `session_id` | `str` | — | Logical conversation identifier |
| `memory_key` | `str` | `"history"` | Key injected into chain input dict |
| `return_messages` | `bool` | `True` | Return list of messages or formatted string |
| `channel` | `str` | `"chat"` | Cortex channel name |
| `salience` | `float` | `0.6` | Memory salience score (0–1) |

## Running Tests

```bash
# Unit tests (no Cortex install required)
pytest tests/unit_tests/ -v

# Integration tests (requires cortex-ai-memory)
pip install cortex-ai-memory
pytest tests/integration_tests/ -v
```

## License

MIT
