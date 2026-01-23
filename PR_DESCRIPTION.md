# PR Title

```
feat(meshtastic): add langchain-meshtastic - let your AI agents talk to the real world via LoRa radio
```

---

## Summary

**Your LangChain agent just got a radio.** This PR adds `langchain-meshtastic`, a new partner package enabling agents to broadcast messages over LoRa mesh networks - no internet required.

- New partner package at `libs/partners/meshtastic`
- `MeshtasticSendTool` for sending messages to decentralized mesh networks
- Auto-detects connected devices or accepts explicit device path configuration  
- Comprehensive unit tests with hardware mocking for CI compatibility

## Why This is Cool

Imagine an AI that can communicate when the grid goes dark. Meshtastic devices use LoRa (Long Range) radio to create peer-to-peer mesh networks that work independently of internet infrastructure - up to several kilometers range with zero monthly fees.

**Use Cases:**
1. **Disaster Response AI**: Deploy agents that coordinate emergency responses when cell towers are down
2. **Off-Grid Operations**: Run AI assistants in remote wilderness, underground facilities, or maritime environments
3. **Resilient Infrastructure**: Build systems that degrade gracefully when connectivity fails
4. **Hardware-AI Bridge**: First step toward agents that interact with the physical world beyond APIs

## Example Usage

```python
from langchain_meshtastic import MeshtasticSendTool
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI

# Initialize the tool (auto-detects connected Meshtastic device)
meshtastic_tool = MeshtasticSendTool()

# Or specify device path explicitly
meshtastic_tool = MeshtasticSendTool(device_path="/dev/ttyUSB0")

# Use with an agent
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools=[meshtastic_tool], ...)

# The agent can now send messages to the mesh network
agent.invoke({"input": "Send an emergency broadcast to the mesh network"})
```

## Dependencies

- `meshtastic` (optional) - Official Python library for Meshtastic devices
  - Only required at runtime when the tool is actually used
  - Install with: `pip install meshtastic`

## Testing

Unit tests mock the hardware interface to ensure CI/CD compatibility without requiring physical devices. Tests cover:

- Tool initialization and schema validation
- Successful message sending
- Error handling (missing library, device not found, permission issues)
- Resource cleanup (interface closure)

## Areas Requiring Careful Review

1. **Package structure**: Follows the same pattern as other partner packages (exa, ollama, etc.)
2. **Error message clarity**: Error messages designed to help users troubleshoot connection issues
3. **Resource cleanup**: `finally` block ensures serial interface is always closed

---

**Note**: This contribution was developed with AI assistance (Claude).
