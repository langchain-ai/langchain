# SuperlinkedRetriever

A LangChain retriever for [Superlinked](https://superlinked.com), a library for building context-aware search and retrieval systems with multi-modal embedding spaces.

## 🚀 Quick Start

### Installation

```bash
pip install -U langchain-superlinked superlinked
```

### Basic Usage

```python
import superlinked.framework as sl
from langchain_superlinked import SuperlinkedRetriever

# 1. Define your data schema
class DocumentSchema(sl.Schema):
    id: sl.IdField
    content: sl.String

doc_schema = DocumentSchema()

# 2. Create semantic spaces
text_space = sl.TextSimilaritySpace(
    text=doc_schema.content,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Build an index
doc_index = sl.Index([text_space])

# 4. Define your search query
query = (
    sl.Query(doc_index)
    .find(doc_schema)
    .similar(text_space.text, sl.Param("query_text"))
    .select([doc_schema.content])
    .limit(sl.Param("limit"))
)

# 5. Set up your data and app
documents = [
    {"id": "1", "content": "Machine learning algorithms process data efficiently."},
    {"id": "2", "content": "Natural language processing understands human language."},
    {"id": "3", "content": "Deep learning requires computational resources."}
]

source = sl.InMemorySource(schema=doc_schema)
executor = sl.InMemoryExecutor(sources=[source], indices=[doc_index])
app = executor.run()
source.put(documents)

# 6. Create the retriever
retriever = SuperlinkedRetriever(
    sl_client=app,
    sl_query=query,
    page_content_field="content",
    k=4  # Number of documents to return (optional, defaults to 4)
)

# 7. Search!
docs = retriever.invoke("artificial intelligence")
# Returns: [Document(page_content="Machine learning...", metadata={"id": "1"}), ...]
```

## 🎯 Key Features

- **🔄 Vector Database Agnostic**: Works with Qdrant, Redis, MongoDB, or in-memory
- **🎛️ Multi-Modal Search**: Combine text, categorical, numerical, temporal, and custom spaces
- **⚡ Runtime Control**: Dynamic weighting and parameter adjustment at query time
- **🧩 LangChain Integration**: Seamless integration with LangChain RAG pipelines
- **📊 Flexible Schemas**: Support for any data structure via Superlinked schemas

## 📋 Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sl_client` | `sl.App` | Required | Superlinked app instance |
| `sl_query` | `QueryDescriptor` | Required | Pre-built Superlinked query |
| `page_content_field` | `str` | Required | Field to use as Document content |
| `k` | `int` | `4` | Number of documents to return |
| `query_text_param` | `str` | `"query_text"` | Parameter name for user query |
| `metadata_fields` | `List[str]` | `None` | Fields to include in metadata |

## 🎨 Advanced Usage Examples

### Multi-Space Blog Search

Combine content similarity, category matching, and recency:

```python
# Define multiple spaces
content_space = sl.TextSimilaritySpace(text=blog_schema.content)
category_space = sl.CategoricalSimilaritySpace(
    category_input=blog_schema.category,
    categories=["tech", "science", "business"]
)
recency_space = sl.RecencySpace(timestamp=blog_schema.published_date)

# Weighted query
blog_query = sl.Query(
    blog_index,
    weights={
        content_space: sl.Param("content_weight"),
        category_space: sl.Param("category_weight"),
        recency_space: sl.Param("recency_weight")
    }
).find(blog_schema).similar(content_space.text, sl.Param("query_text"))

# Dynamic weighting at runtime
results = retriever.invoke(
    "machine learning",
    content_weight=1.0,    # Prioritize content similarity
    category_weight=0.3,   # Some category influence
    recency_weight=0.8,    # Favor recent posts
    k=5
)
```

### E-commerce Product Search

Balance text similarity, price preferences, and ratings:

```python
# Spaces for different aspects
description_space = sl.TextSimilaritySpace(text=product_schema.description)
price_space = sl.NumberSpace(
    number=product_schema.price,
    mode=sl.Mode.MINIMUM  # Favor lower prices
)
rating_space = sl.NumberSpace(
    number=product_schema.rating,
    mode=sl.Mode.MAXIMUM  # Favor higher ratings
)

# Search strategies
quality_focused = retriever.invoke(
    "wireless headphones",
    description_weight=0.7,
    price_weight=0.1,
    rating_weight=1.0,  # Prioritize quality
    k=3
)

budget_focused = retriever.invoke(
    "wireless headphones",
    description_weight=0.6,
    price_weight=1.0,   # Prioritize low price
    rating_weight=0.3,
    k=3
)
```

### Production Deployment with Qdrant

Switch to persistent vector storage without changing retriever code:

```python
# Configure Qdrant
qdrant_db = sl.QdrantVectorDatabase(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
    vector_precision=sl.Precision.FLOAT16
)

# Same setup, different storage
executor = sl.InMemoryExecutor(
    sources=[source],
    indices=[doc_index],
    vector_database=qdrant_db  # 👈 Only this line changes!
)

# Same retriever code works unchanged
retriever = SuperlinkedRetriever(
    sl_client=app,
    sl_query=query,
    page_content_field="content"
)
```

## 🔗 LangChain Integration

### Basic RAG Pipeline

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer based on context:

Context: {context}
Question: {question}
""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
)

# Use it
answer = chain.invoke("How does machine learning work?")
```

### Dynamic Parameter Control

```python
# Override k at query time
few_results = retriever.invoke("AI trends", k=2)
many_results = retriever.invoke("AI trends", k=10)

# Custom parameters for different scenarios
technical_docs = retriever.invoke(
    "API documentation",
    category_filter="technical",
    limit=5
)
```

## 🏗️ Architecture Benefits

### Vector Database Agnostic Design

The same `SuperlinkedRetriever` code works with any vector database:

```python
# In-Memory (development)
executor = sl.InMemoryExecutor(sources=[source], indices=[index])

# Qdrant (production)
executor = sl.InMemoryExecutor(
    sources=[source],
    indices=[index],
    vector_database=sl.QdrantVectorDatabase(...)
)

# Redis (caching layer)
executor = sl.InMemoryExecutor(
    sources=[source],
    indices=[index],
    vector_database=sl.RedisVectorDatabase(...)
)

# MongoDB (document storage)
executor = sl.InMemoryExecutor(
    sources=[source],
    indices=[index],
    vector_database=sl.MongoDbVectorDatabase(...)
)
```

### Multi-Modal Search Capabilities

| Space Type | Use Case | Example |
|------------|----------|---------|
| `TextSimilaritySpace` | Semantic text search | Content, titles, descriptions |
| `CategoricalSimilaritySpace` | Category matching | Topics, brands, types |
| `RecencySpace` | Time-based relevance | Recent posts, fresh content |
| `NumberSpace` | Numerical preferences | Prices, ratings, scores |
| `CustomSpace` | Domain-specific logic | Business rules, custom algorithms |

## 🎛️ Runtime Parameter Control

All Superlinked parameters can be controlled at query time:

```python
# Base retriever setup
retriever = SuperlinkedRetriever(
    sl_client=app,
    sl_query=weighted_query,
    page_content_field="content"
)

# Different search strategies with same retriever
content_focused = retriever.invoke(
    "search term",
    content_weight=1.0,
    recency_weight=0.2,
    k=5
)

recency_focused = retriever.invoke(
    "search term",
    content_weight=0.5,
    recency_weight=1.0,
    k=3
)

balanced_search = retriever.invoke(
    "search term",
    content_weight=0.7,
    category_weight=0.5,
    recency_weight=0.6,
    popularity_weight=0.4,
    k=8
)
```

## 📚 Example Files

This repository includes comprehensive examples:

- **[`langchain_retreiver.py`](./langchain_retreiver.py)**: Complete SuperlinkedRetriever implementation
- **[`superlinked_retriever_examples.py`](./superlinked_retriever_examples.py)**: Six detailed usage examples:
  1. **Simple Text Search**: Basic semantic similarity
  2. **Multi-Space Blog Search**: Content + category + recency + popularity
  3. **E-commerce Product Search**: Description + price + ratings + brand
  4. **News Article Search**: Content + sentiment + topics + recency
  5. **LangChain RAG Integration**: Complete RAG pipeline example
  6. **Qdrant Vector Database**: Production deployment example

## 🔧 Troubleshooting

### Common Issues

1. **"sl_query must be a Superlinked QueryDescriptor instance"**
   - Use `sl.Query(...).find(...).similar(...)` (returns `QueryDescriptor`)
   - Not `sl.Query(...)` alone (returns `Query`)

2. **"InteractiveSource.__init__() got an unexpected keyword argument 'data'"**
   - Create source first: `source = sl.InMemorySource(schema=schema)`
   - Run executor: `app = executor.run()`
   - Then add data: `source.put(data)`

3. **"'<=' not supported between instances of 'int' and 'datetime.datetime'"**
   - Use Unix timestamps for `sl.Timestamp` fields
   - Convert: `int(datetime.now().timestamp())`

4. **"Unindexed fields with filter found"**
   - Remove `.filter()` clauses if fields aren't properly indexed
   - Or ensure filtered fields are included in the index definition

### Performance Tips

- Use `vector_precision=sl.Precision.FLOAT16` for Qdrant to save memory
- Set appropriate `default_query_limit` for your use case
- Consider caching strategies for frequently accessed data
- Use appropriate `k` values to balance relevance and performance

## 🏃‍♂️ Getting Started Quickly

1. **Copy the basic example** from the Quick Start section
2. **Modify the schema** to match your data structure
3. **Add relevant spaces** for your use case (text, categorical, numerical, etc.)
4. **Adjust parameters** at runtime for different search strategies
5. **Integrate with LangChain** for complete RAG pipelines

## 🌟 Use Cases

The SuperlinkedRetriever excels in scenarios requiring:

- **Multi-faceted search**: Combining text similarity with other criteria
- **Dynamic ranking**: Runtime control over search priorities
- **Domain-specific retrieval**: Custom spaces for business logic
- **Production deployment**: Vector database flexibility
- **RAG applications**: Context-aware document retrieval

## 📈 Scaling Considerations

### Development → Production

```python
# Development: Quick setup with in-memory storage
source = sl.InMemorySource(schema=schema)
executor = sl.InMemoryExecutor(sources=[source], indices=[index])

# Production: Persistent storage with Qdrant/Redis/MongoDB
source = sl.InMemorySource(schema=schema)  # Same source
executor = sl.InMemoryExecutor(
    sources=[source],
    indices=[index],
    vector_database=sl.QdrantVectorDatabase(...)  # Only this changes
)
```

### Performance Optimization

- **Batch data ingestion** for large datasets
- **Appropriate vector precision** for storage efficiency
- **Query result caching** for repeated searches
- **Index optimization** for frequently searched fields

## 🤝 Contributing

The SuperlinkedRetriever is designed to be:
- **Flexible**: Works with any Superlinked schema and space configuration
- **Consistent**: Standard LangChain retriever interface
- **Extensible**: Easy to add new parameter types and configurations
- **Production-Ready**: Vector database agnostic for deployment flexibility

## 📚 Learn More

- [Superlinked Repo](https://github.com/superlinked/superlinked)
- [LangChain Retrievers Guide](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Vector Database Comparison](https://docs.superlinked.com/run-in-production/vdbs)
- [Superlinked Concepts](https://docs.superlinked.com/concepts/features)

---
