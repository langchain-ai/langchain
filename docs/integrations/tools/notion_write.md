# Notion Write Toolkit

## Overview

The Notion Write Toolkit exposes LangChain-ready tools for creating and updating Notion pages via the official Notion API. It bundles search and write utilities with shared client configuration so agents can draft meeting notes, status updates, or decision logs directly inside Notion workspaces.

## Installation

```bash
pip install langchain-notion-tools
```

## Example usage

```python
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_notion_tools import create_toolkit

notion_toolkit = create_toolkit()
notion_tools = list(notion_toolkit.tools)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent(
    notion_tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

response = agent.invoke(
    {
        "input": (
            "Create a Notion page titled 'LLM Launch Update' under the default parent "
            "and summarise today's release milestones as bullet points."
        )
    }
)
print(response["output"])
```

## Live demo

![Notion page creation demo](./notion-write-demo.png)

## Resources

- [GitHub repository](https://github.com/dineshkumarkummara/langchain-notion-tool)
- [PyPI package](https://pypi.org/project/langchain-notion-tools/)
