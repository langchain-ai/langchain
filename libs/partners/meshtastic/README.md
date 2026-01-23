# langchain-meshtastic

This package provides LangChain integrations for [Meshtastic](https://meshtastic.org/) LoRa mesh network devices.

## Overview

Meshtastic is an open-source project that enables long-range, low-power communication using LoRa radio technology. This integration allows LangChain agents to send messages through Meshtastic mesh networks, enabling AI communication in offline or infrastructure-limited environments.

## Installation

```bash
pip install langchain-meshtastic
```

You'll also need the Meshtastic library and a connected Meshtastic device:

```bash
pip install meshtastic
```

## Usage

### Basic Usage

```python
from langchain_meshtastic import MeshtasticSendTool

# Auto-detect connected Meshtastic device
tool = MeshtasticSendTool()

# Send a message to the mesh network
result = tool.invoke({"message": "Hello from AI!", "channel_index": 0})
print(result)
```

### With Explicit Device Path

```python
from langchain_meshtastic import MeshtasticSendTool

# Specify device path (Linux/macOS)
tool = MeshtasticSendTool(device_path="/dev/ttyUSB0")

# Or on Windows
tool = MeshtasticSendTool(device_path="COM3")
```

### With LangChain Agents

```python
from langchain_meshtastic import MeshtasticSendTool
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI

# Initialize the tool
meshtastic_tool = MeshtasticSendTool()

# Create an agent with the tool
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools=[meshtastic_tool], ...)

# The agent can now send messages to the mesh network
agent.invoke({"input": "Send an emergency broadcast to the mesh network"})
```

## Use Cases

- **Disaster Response**: Deploy AI agents that can communicate when cellular/internet infrastructure is down
- **Remote Operations**: Enable AI assistants in wilderness, maritime, or underground environments
- **Resilient Systems**: Build applications that gracefully degrade when connectivity fails
- **Off-Grid IoT**: Bridge LLMs with physical hardware for offline-first applications

## Hardware Requirements

- A Meshtastic-compatible device (e.g., LILYGO T-Beam, Heltec LoRa 32, RAK WisBlock)
- USB connection to your computer
- Meshtastic firmware installed on the device

## Resources

- [Meshtastic Documentation](https://meshtastic.org/docs/)
- [Supported Hardware](https://meshtastic.org/docs/hardware)
- [LangChain Documentation](https://docs.langchain.com/)
