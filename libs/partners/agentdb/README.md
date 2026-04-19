# langchain-agentdb

This package provides a LangChain integration for [AgentDB](https://agentdb.dev) —
a real-time curated knowledge API for AI agents.

AgentDB ingests YouTube channels, podcasts, and blog feeds on a Mon/Wed/Fri schedule,
summarises each item with Claude, and exposes the results via a structured REST API
with optional semantic vector search.

## Installation

```bash
pip install -U langchain-agentdb
```

## Setup

Sign up for a free API key at [agentdb.dev](https://agentdb.dev):

```bash
curl -X POST https://agentdb-production-9ba0.up.railway.app/v1/auth/register \
     -H "Content-Type: application/json" -d '{}'
```

```bash
export AGENTDB_API_KEY="agentdb-..."
```

## Usage

```python
from langchain_agentdb import AgentDBRetriever

# Semantic search (Pro tier)
retriever = AgentDBRetriever(mode="search", k=10)
docs = retriever.invoke("AI safety news this week")

# Latest items (free tier)
retriever = AgentDBRetriever(mode="latest", k=20, content_type="podcast")
docs = retriever.invoke("")

for doc in docs:
    print(doc.metadata["title"])
    print(doc.page_content[:300])
```

## In a chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

retriever = AgentDBRetriever(k=5)
llm = ChatOpenAI(model="gpt-4o-mini")

chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(
        "Answer using this knowledge:\n{context}\n\nQuestion: {question}"
    )
    | llm
)

answer = chain.invoke("What's the latest in AI?")
```

## Sources

AgentDB ingests 20 curated sources across market news, technology/AI, and science/philosophy,
updated Monday, Wednesday, and Friday at 07:00 UTC.

See [agentdb.dev/sources](https://agentdb.dev/sources) for the full list.
