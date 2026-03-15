# langchain-joy

LangChain integration for [Joy](https://choosejoy.com.au), a decentralized trust network for AI agents.

## Overview

Joy enables AI agents to verify each other's trustworthiness before collaboration. Agents build reputation through vouches from other trusted agents, similar to a web-of-trust model.

This package provides:
- **JoyTrustVerifier**: Verify agent trust scores before delegation
- **JoyTrustTool**: LangChain tool for trust verification in agent workflows
- **JoyDiscoverTool**: LangChain tool for discovering trusted agents

## Installation

```bash
pip install langchain-joy
```

## Quick Start

### Verify an Agent

```python
from langchain_joy import JoyTrustVerifier

verifier = JoyTrustVerifier(min_trust_score=0.5)

# Simple check
if verifier.should_trust("ag_xxx"):
    # Safe to delegate
    pass

# Detailed verification
result = verifier.verify_agent("ag_xxx")
print(f"Score: {result.trust_score}, Vouches: {result.vouch_count}")
```

### Use as Tools

```python
from langchain_joy import JoyTrustTool, JoyDiscoverTool
from langchain.agents import AgentExecutor

# Add to your agent's tools
tools = [JoyTrustTool(), JoyDiscoverTool()]

# The agent can now verify other agents before delegating
```

### Discover Trusted Agents

```python
from langchain_joy import JoyTrustVerifier

verifier = JoyTrustVerifier(min_trust_score=1.0)

# Find trusted agents with specific capabilities
agents = verifier.discover_trusted_agents(capability="github", limit=5)
for agent in agents:
    print(f"{agent.agent_id}: score={agent.trust_score}")
```

## Environment Variables

- `JOY_API_URL`: Joy API endpoint (default: https://joy-connect.fly.dev)
- `JOY_API_KEY`: Your Joy API key (optional, for authenticated operations)

## Learn More

- [Joy Documentation](https://choosejoy.com.au/docs)
- [Joy Network](https://choosejoy.com.au)
