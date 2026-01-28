# langchain-chroma

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-chroma?label=%20)](https://pypi.org/project/langchain-chroma/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-chroma)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-chroma)](https://pypistats.org/packages/langchain-chroma)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-chroma
```

## 🤔 What is this?

This package contains the LangChain integration with Chroma.

## 📖 Documentation

View the [documentation](https://docs.langchain.com/oss/python/integrations/providers/chroma) for more details.

## 🚀 Example: RAG with Metadata Filtering

Here's a complete example of building a RAG (Retrieval-Augmented Generation) system with Chroma that uses metadata filtering for precise document retrieval:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Create documents with metadata
documents = [
    Document(
        page_content="Python is a high-level programming language.",
        metadata={"category": "programming", "language": "python", "year": 2024}
    ),
    Document(
        page_content="JavaScript is used for web development.",
        metadata={"category": "programming", "language": "javascript", "year": 2024}
    ),
    Document(
        page_content="Machine learning models require large datasets.",
        metadata={"category": "ai", "language": "python", "year": 2023}
    ),
    Document(
        page_content="Vector databases enable semantic search.",
        metadata={"category": "ai", "language": "python", "year": 2024}
    ),
]

# 2. Initialize Chroma vector store with embeddings
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="rag_example"
)

# 3. Create a retriever with metadata filtering
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 2,
        "filter": {"category": "programming"}  # Filter by metadata
    }
)

# 4. Build RAG chain
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the provided context. Include source citations."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def format_docs(docs):
    return "\n\n".join([
        f"[Source: {doc.metadata.get('language', 'unknown')}] {doc.page_content}"
        for doc in docs
    ])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Query with metadata filtering
response = rag_chain.invoke("What programming languages are mentioned?")
print(response)
```

### Advanced: Dynamic Metadata Filtering

You can also create a retriever that accepts dynamic filters:

```python
from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any

class FilteredRetriever(BaseRetriever):
    def __init__(self, vectorstore: Chroma, default_filter: Dict[str, Any] = None):
        self.vectorstore = vectorstore
        self.default_filter = default_filter or {}
    
    def _get_relevant_documents(self, query: str, *, filter: Dict[str, Any] = None):
        search_filter = {**self.default_filter, **(filter or {})}
        return self.vectorstore.similarity_search(
            query,
            k=3,
            filter=search_filter
        )

# Use with dynamic filters
retriever = FilteredRetriever(vectorstore, default_filter={"year": 2024})
results = retriever._get_relevant_documents(
    "What is Python?",
    filter={"category": "programming"}
)
```

### Key Features Demonstrated

- ✅ **Metadata filtering**: Filter documents by category, language, year, etc.
- ✅ **Source citations**: Include metadata in responses for traceability
- ✅ **Flexible retrieval**: Combine semantic search with metadata constraints
- ✅ **Production-ready**: Uses LangChain's standard patterns

This example shows how to build production RAG systems that combine semantic search with structured metadata filtering for precise, context-aware retrieval.
