# langchain-decodo

LangChain partner package for the [Decodo](https://decodo.com) web scraping API.

Provides two **LangChain tools** and a **document loader** that let LLM agents
and RAG pipelines fetch live web content without managing proxies, JavaScript
rendering, or anti-bot protection.

## Installation

```bash
pip install langchain-decodo
```

## Authentication

All classes read your Decodo API token from the `DECODO_API_TOKEN` environment
variable, or you can pass it explicitly.

```bash
export DECODO_API_TOKEN="your-decodo-api-token"
```

Get a token from the [Decodo Dashboard](https://app.decodo.com).

## Components

### `DecodoWebScrapeTool`

Scrape any URL and return its full content as markdown or plain text.
Handles JavaScript rendering, CAPTCHAs, and geo-blocking automatically.

```python
from langchain_decodo import DecodoWebScrapeTool

tool = DecodoWebScrapeTool()  # reads DECODO_API_TOKEN from env
content = tool.run("https://example.com")
print(content)
```

Pass explicitly:

```python
from pydantic import SecretStr
from langchain_decodo import DecodoWebScrapeTool

tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("YOUR_TOKEN"))
```

### `DecodoSearchTool`

Search Google, Amazon, or Reddit and return structured JSON results.

```python
from langchain_decodo import DecodoSearchTool

tool = DecodoSearchTool()

# Google search (default)
results = tool.run({"query": "LangChain latest release", "engine": "google"})

# Amazon product search
results = tool.run({"query": "Python programming book", "engine": "amazon"})

# Reddit discussion search
results = tool.run({"query": "best web scraping libraries", "engine": "reddit"})
```

Returns a JSON string — a list of objects with `content`, `url`, and
`status_code` fields.

Supported engines:

| `engine` | Decodo target | Description |
|---|---|---|
| `google` | `google_search` | Google SERP |
| `amazon` | `amazon_search` | Amazon product search |
| `reddit` | `google_search` + `site:reddit.com` | Reddit via Google |

### `DecodoLoader`

Load one or more URLs as LangChain `Document` objects for use in RAG pipelines.

```python
from langchain_decodo import DecodoLoader

loader = DecodoLoader(
    urls=[
        "https://python.org/about/",
        "https://docs.python.org/3/whatsnew/3.12.html",
    ],
)
docs = loader.load()

for doc in docs:
    print(doc.metadata["url"], "—", len(doc.page_content), "chars")
```

Each `Document` has:

- `page_content` — scraped text/markdown.
- `metadata["url"]` — the source URL.
- `metadata["source"]` — same as `url` (LangChain convention).
- `metadata["status_code"]` — HTTP status from the target site.

## LangChain agent example

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_decodo import DecodoWebScrapeTool, DecodoSearchTool

tools = [DecodoWebScrapeTool(), DecodoSearchTool()]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({
    "input": "What is the latest stable version of Python? Check python.org."
})
print(result["output"])
```

## RAG pipeline example

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_decodo import DecodoLoader

loader = DecodoLoader(urls=["https://python.org/about/"])
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

store = FAISS.from_documents(chunks, OpenAIEmbeddings())
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=store.as_retriever(search_kwargs={"k": 4}),
)

result = chain.invoke({"query": "What is Python used for?"})
print(result["result"])
```

## Links

- [Decodo website](https://decodo.com)
- [Decodo API documentation](https://developers.decodo.com)
- [Decodo Dashboard](https://app.decodo.com)
- [LangChain documentation](https://python.langchain.com)
