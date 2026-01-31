# Cost-Aware & Traced RAG Pipeline

This guide shows how to build a production-ready RAG system with:

- Token usage tracking
- Cost estimation
- Optional OpenTelemetry tracing

## Why this matters
Most RAG examples ignore:
- Cost overruns
- Debugging difficulty
- Production observability

This guide fixes that by introducing a `CostTrackingCallback` and integrating OpenTelemetry.

## Setup

Ensure you have the necessary packages installed:

```bash
pip install langchain langchain-openai faiss-cpu opentelemetry-api
```

## Implementation

### 1. Cost Tracking Callback

We use a custom `CostTrackingCallback` to monitor token usage and calculate costs.

```python
from langchain_classic.callbacks.cost_tracking import CostTrackingCallback

# Initialize with your specific cost per 1k tokens (e.g., $0.002 for GPT-3.5-Turbo)
cost_callback = CostTrackingCallback(cost_per_1k_tokens=0.002)
```

### 2. Building the RAG Pipeline

Here is the full example of how to integrate this into a RAG pipeline.

```python
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager

# 1. Setup Data
docs = ["LangChain is great.", "Observability is key."]
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = splitter.create_documents(docs)

# 2. Vector Store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

# 3. Configure Callback
cost_callback = CostTrackingCallback(cost_per_1k_tokens=0.002)
callback_manager = CallbackManager([cost_callback])

# 4. Initialize LLM with Callback
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    callback_manager=callback_manager
)

# 5. Create Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# 6. Run & Inspect
query = "Why is observability important?"
qa.run(query)

print(cost_callback.summary())
# Output: {'total_tokens': 150, 'total_cost_usd': 0.0003, 'latency_sec': 1.2}
```

## Adding Tracing (OpenTelemetry)

You can wrap your chain execution in an OpenTelemetry span for deeper observability.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("rag-query"):
    qa.run(query)
```

This ensures that your RAG pipeline is not only cost-aware but also fully traceable in your observability platform.
