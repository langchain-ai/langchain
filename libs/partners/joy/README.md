# langchain-joy

Joy Trust Network integration for LangChain - verify agent trust scores before delegation.

## Installation

```bash
pip install langchain-joy
```

## Quick Start

Add trust verification to any LangChain agent with a callback handler:

```python
from langchain_joy import JoyTrustCallbackHandler
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Create callback handler with trust threshold
handler = JoyTrustCallbackHandler(min_trust_score=1.5)

# Add to any agent
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[handler],
)

# Tool calls are now verified against Joy trust scores
result = agent.run("Use the calculator tool to compute 2+2")
```

## How It Works

The callback handler intercepts `on_tool_start` events and verifies the tool/agent against Joy's trust network before allowing execution:

1. Extracts agent ID from tool metadata or name
2. Queries Joy API for trust score
3. Blocks execution if score < threshold (fail-closed)
4. Logs all verification attempts for audit

## Configuration

```python
handler = JoyTrustCallbackHandler(
    min_trust_score=1.5,      # Minimum trust score (0-5 scale)
    fail_open=False,          # Block on errors (default: False)
    api_key="joy_xxx",        # Optional API key for higher rate limits
    cache_ttl=300,            # Cache trust scores for 5 minutes
)
```

### Recommended Thresholds

| Level | Score | Use Case |
|-------|-------|----------|
| permissive | 1.0 | Low-risk tasks, broad agent discovery |
| standard | 1.5 | General use (recommended default) |
| moderate | 2.0 | Established agents only |
| strict | 2.5 | High security, top-tier agents |

## Verification Decorator

For more control, use the decorator on specific functions:

```python
from langchain_joy import require_trust

@require_trust(min_score=2.0, agent_id_param="target_agent")
def delegate_to_agent(target_agent: str, task: str) -> str:
    # Only executes if target_agent has trust >= 2.0
    return external_agent.run(task)
```

## Direct API Access

```python
from langchain_joy import JoyTrustClient

client = JoyTrustClient()

# Check trust score
result = client.get_trust_score("ag_abc123")
print(f"Trust: {result['trust_score']}, Verified: {result['verified']}")

# Discover trusted agents
agents = client.discover_agents(capability="code-review", min_trust=1.5)
```

## Links

- [Joy Trust Network](https://choosejoy.com.au)
- [API Documentation](https://choosejoy.com.au/docs)
- [GitHub](https://github.com/tlkc888-Jenkins/Joy)
