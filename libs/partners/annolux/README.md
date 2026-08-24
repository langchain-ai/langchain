# LangChain Annolux Integration

Official [LangChain](https://langchain.com) integration for **[Annolux](https://annolux.com)** — The curated English & Chinese web search API and MCP for AI Agents with explicit snapshot timestamps.

[![PyPI version](https://img.shields.io/pypi/v/langchain-annolux.svg?style=flat-square)](https://pypi.org/project/langchain-annolux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## ⚡ Installation

```bash
pip install langchain-annolux
```

---

## 🔑 Configuration

Get your free API key (1,000 requests/month) at [annolux.com](https://annolux.com) and set it in your environment:

```bash
export ANNOLUX_API_KEY="ann_live_your_key_here"
```

---

## 🚀 Quickstart

### 1. Standalone Tool Execution
```python
from langchain_annolux import AnnoluxSearchRun

search = AnnoluxSearchRun()
print(search.run("Latest DeepSeek R1 reinforcement learning architecture"))
```

### 2. Structured JSON Output
```python
from langchain_annolux import AnnoluxSearchResults

tool = AnnoluxSearchResults()
results = tool.invoke({"query": "Go 1.26 release notes", "max_results": 3})
for r in results:
    print(f"[{r['fetched_at']}] {r['title']} -> {r['url']}")
```

### 3. Binding to LangChain Agents
```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_annolux import AnnoluxSearchRun

llm = ChatOpenAI(temperature=0, model="gpt-4o")
tools = [AnnoluxSearchRun()]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

agent.run("What are the key differences between vLLM and TensorRT-LLM in 2026?")
```

---

## 📄 License

MIT © [Annolux](https://annolux.com)
