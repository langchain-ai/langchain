# langchain-plasmate

LangChain integration for [Plasmate](https://github.com/plasmate-labs/plasmate) - browse the web with 10x fewer tokens using SOM (Semantic Object Model).

## What is Plasmate?

Plasmate is a headless browser engine built for AI agents. Instead of raw HTML, it outputs SOM - a structured JSON format that preserves content, structure, and interactivity while discarding presentation markup.

**Benchmarks across 49 real-world websites:**
- 16.6x overall token compression
- 10.5x median compression
- 94% cost savings at GPT-4/GPT-4o/Claude pricing

## Installation

```bash
pip install langchain-plasmate
```

You also need the Plasmate binary:

```bash
cargo install plasmate
# or
curl -fsSL https://plasmate.app/install.sh | sh
```

## Tools

### PlasmateFetchTool

Fetch a web page and return structured content:

```python
from langchain_plasmate import PlasmateFetchTool

tool = PlasmateFetchTool()
result = tool.invoke({"url": "https://news.ycombinator.com"})
print(result)
```

### PlasmateNavigateTool

Navigate to a page with interactive element details:

```python
from langchain_plasmate import PlasmateNavigateTool

tool = PlasmateNavigateTool()
result = tool.invoke({"url": "https://example.com", "extract_links": True})
```

### Use with an Agent

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_plasmate import PlasmateFetchTool, PlasmateNavigateTool

tools = [PlasmateFetchTool(), PlasmateNavigateTool()]
llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You can browse the web using Plasmate tools. Pages are returned in a structured, token-efficient format."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "What are the top stories on Hacker News?"})
```

## Document Loader

Load web pages as LangChain documents:

```python
from langchain_plasmate import PlasmateLoader

loader = PlasmateLoader(urls=[
    "https://example.com",
    "https://news.ycombinator.com",
])

docs = loader.load()
for doc in docs:
    print(f"{doc.metadata['title']} ({doc.metadata['compression_ratio']}x compression)")
    print(doc.page_content[:200])
```

## Why SOM instead of raw HTML?

| Site | HTML Tokens | SOM Tokens | Savings |
|------|------------|------------|---------|
| github.com | 94,956 | 9,005 | 90% |
| vercel.com | 198,761 | 5,565 | 97% |
| cloud.google.com | 464,616 | 3,973 | 99% |
| reuters.com | 262,746 | 19,586 | 93% |

Full benchmark: [plasmate.app/docs/benchmark-cost](https://plasmate.app/docs/benchmark-cost)

## Resources

- [Plasmate GitHub](https://github.com/plasmate-labs/plasmate) (Apache 2.0)
- [SOM Spec v1.0](https://plasmate.app/docs/som-spec)
- [Why SOM](https://plasmate.app/docs/why-som)
- [Cost Analysis](https://plasmate.app/docs/benchmark-cost)
