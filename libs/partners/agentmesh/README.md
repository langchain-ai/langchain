# langchain-agentmesh

AgentMesh trust layer integration for LangChain - enabling cryptographic identity verification and trust-gated tool execution for LLM agents.

## Overview

This package provides:

- **Trusted Tool Execution**: Verify agent identity before allowing tool calls
- **Trust-Gated Callbacks**: Monitor and enforce trust policies during chain execution
- **Cryptographic Identity**: CMVK (Cryptographic Multi-Vector Keys) based agent authentication
- **Agent Handoff Verification**: Secure multi-agent workflows with verified handoffs

## Installation

```bash
pip install langchain-agentmesh
```

Or with poetry:

```bash
poetry add langchain-agentmesh
```

## Quick Start

### Creating a Trusted Agent Identity

```python
from langchain_agentmesh import CMVKIdentity, TrustedToolExecutor

# Generate cryptographic identity for your agent
identity = CMVKIdentity.generate(
    agent_name="research-agent",
    capabilities=["web_search", "document_analysis"]
)

# Create trust-gated tool executor
executor = TrustedToolExecutor(identity=identity)
```

### Trust-Gated Tool Execution

```python
from langchain_agentmesh import TrustGatedTool, TrustPolicy
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the internal database."""
    return f"Results for: {query}"

# Wrap tool with trust verification
trusted_tool = TrustGatedTool(
    tool=search_database,
    required_capabilities=["database_access"],
    min_trust_score=0.8
)

# Execute with identity verification
result = executor.invoke(trusted_tool, "financial reports 2024")
```

### Trust Callback Handler

```python
from langchain_agentmesh import TrustCallbackHandler
from langchain_openai import ChatOpenAI

# Add trust monitoring to your chain
trust_handler = TrustCallbackHandler(
    identity=identity,
    policy=TrustPolicy(
        require_verification=True,
        min_trust_score=0.7,
        audit_all_calls=True
    )
)

llm = ChatOpenAI(callbacks=[trust_handler])
```

### Multi-Agent Trust Handoffs

```python
from langchain_agentmesh import TrustHandshake, TrustedAgentCard

# Create agent card for discovery
agent_card = TrustedAgentCard(
    name="research-agent",
    description="Performs web research and analysis",
    capabilities=["web_search", "summarization"],
    identity=identity
)
agent_card.sign(identity)

# Verify another agent before handoff
handshake = TrustHandshake(my_identity=identity)
peer_card = TrustedAgentCard.from_json(peer_card_json)

verification = handshake.verify_peer(peer_card, min_trust_score=0.8)
if verification.trusted:
    # Safe to hand off task
    pass
```

## Features

### Trust Policies

Configure trust requirements for your agents:

```python
from langchain_agentmesh import TrustPolicy

policy = TrustPolicy(
    require_verification=True,      # Require identity verification
    min_trust_score=0.8,            # Minimum trust score (0-1)
    allowed_capabilities=["read"],  # Restrict to specific capabilities
    audit_all_calls=True,           # Log all tool invocations
    block_unverified=True           # Block calls from unverified agents
)
```

### Delegation Chains

Support for hierarchical trust delegation:

```python
from langchain_agentmesh import DelegationChain

# Create delegation chain from root authority
chain = DelegationChain(root_identity=admin_identity)

# Delegate capabilities to sub-agents
chain.add_delegation(
    delegatee=worker_agent_card,
    capabilities=["read", "write"],
    expires_in_hours=24
)

# Verify delegation is valid
is_valid = chain.verify()
```

## Integration with LangGraph

For multi-agent workflows with LangGraph:

```python
from langchain_agentmesh import TrustGatedNode
from langgraph.graph import StateGraph

# Create trust-gated nodes
research_node = TrustGatedNode(
    agent=research_agent,
    identity=research_identity,
    required_trust=0.8
)

# Handoffs verify trust automatically
graph = StateGraph()
graph.add_node("research", research_node)
graph.add_edge("research", "analysis", verify_trust=True)
```

## Security Model

AgentMesh uses Ed25519 cryptography for:

- **Identity Generation**: Unique DID (Decentralized Identifier) per agent
- **Message Signing**: All trust assertions are cryptographically signed
- **Verification**: Public key verification without central authority

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `CMVKIdentity` | Cryptographic identity for agents |
| `TrustedToolExecutor` | Execute tools with trust verification |
| `TrustGatedTool` | Wrap tools with trust requirements |
| `TrustCallbackHandler` | LangChain callback for trust monitoring |
| `TrustHandshake` | Verify peer agents |
| `TrustedAgentCard` | Agent discovery and verification card |
| `DelegationChain` | Hierarchical trust delegation |
| `TrustPolicy` | Configure trust requirements |

## License

MIT License - see [LICENSE](LICENSE) for details.
