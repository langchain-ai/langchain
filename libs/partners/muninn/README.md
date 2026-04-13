# langchain-muninn

An integration package connecting Muninn memory and LangChain.

## Installation

```bash
pip install langchain-muninn
```

## Features

- **MuninnMemory**: Conversational memory backed by Muninn's semantic search
- **MuninnEntityMemory**: Entity-focused memory for tracking facts about people, organizations, and concepts
- **99.1% LOCOMO Accuracy**: Highest publicly reported score on the LOCOMO benchmark

## Quick Start

```python
from langchain_muninn import MuninnMemory
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI

# Initialize Muninn memory
memory = MuninnMemory(
    api_key="muninn_xxx",
    organization_id="my-agent"
)

# Use with an agent
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent="zero-shot-react-description"
)
```

## MuninnEntityMemory

For agents that need to remember specific facts:

```python
from langchain_muninn import MuninnEntityMemory

memory = MuninnEntityMemory(
    api_key="muninn_xxx",
    organization_id="my-agent"
)

# Automatically extracts and stores entity facts
# Example: "James works at TechCorp" -> {James: {works_at: TechCorp}}
```

## Links

- [Muninn Documentation](https://muninn.au)
- [LOCOMO Benchmark Results](https://muninn.au/benchmark)
- [GitHub](https://github.com/Phillipneho/muninn)
- [PyPI](https://pypi.org/project/muninn-sdk/)

## License

MIT