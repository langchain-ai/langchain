# Cohere

>[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models
> that help companies improve human-machine interactions.

## Installation and Setup
- Install the Python SDK :
```bash
pip install langchain-cohere
```

Get a [Cohere api key](https://dashboard.cohere.ai/) and set it as an environment variable (`COHERE_API_KEY`)

## Cohere langchain integrations

| API              | description                      | Endpoint docs                                          | Import                                                               | Example usage                                                 |
| ---------------- | -------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------- |
| Chat             | Build chat bots                  | [chat](https://docs.cohere.com/reference/chat)         | `from langchain_cohere import ChatCohere`                            | [cohere.ipynb](/docs/integrations/chat/cohere)                |
| LLM              | Generate text                    | [generate](https://docs.cohere.com/reference/generate) | `from langchain_cohere import Cohere`                                | [cohere.ipynb](/docs/integrations/llms/cohere)                |
| RAG Retriever    | Connect to external data sources | [chat + rag](https://docs.cohere.com/reference/chat)   | `from langchain.retrievers import CohereRagRetriever`                | [cohere.ipynb](/docs/integrations/retrievers/cohere)          |
| Text Embedding   | Embed strings to vectors         | [embed](https://docs.cohere.com/reference/embed)       | `from langchain_cohere import CohereEmbeddings`                      | [cohere.ipynb](/docs/integrations/text_embedding/cohere)      |
| Rerank Retriever | Rank strings based on relevance  | [rerank](https://docs.cohere.com/reference/rerank)     | `from langchain.retrievers.document_compressors import CohereRerank` | [cohere.ipynb](/docs/integrations/retrievers/cohere-reranker) |

## Quick copy examples

### Chat

```python
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
chat = ChatCohere()
messages = [HumanMessage(content="knock knock")]
print(chat(messages))
```

### LLM


```python
from langchain_cohere import Cohere

llm = Cohere(model="command")
print(llm.invoke("Come up with a pet name"))
```

### ReAct Agent

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_cohere import ChatCohere, create_cohere_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor

llm = ChatCohere()

internet_search = TavilySearchResults(max_results=4)
internet_search.name = "internet_search"
internet_search.description = "Route a user query to the internet"

prompt = ChatPromptTemplate.from_template("{input}")

agent = create_cohere_react_agent(
    llm,
    [internet_search],
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=[internet_search], verbose=True)```

agent_executor.invoke({
    "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?",
})
```

### RAG Retriever

```python
from langchain_cohere import ChatCohere
from langchain.retrievers import CohereRagRetriever
from langchain_core.documents import Document

rag = CohereRagRetriever(llm=ChatCohere())
print(rag.get_relevant_documents("What is cohere ai?"))
```

### Text Embedding

```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
print(embeddings.embed_documents(["This is a test document."]))
```
