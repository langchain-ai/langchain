# langchain-gpubridge

LangChain integration for [GPU-Bridge](https://gpubridge.xyz) — 30 AI services (LLM, image, embeddings, STT, TTS, video, PDF) via a single endpoint.

## Installation

```bash
pip install langchain-gpubridge
```

## Setup

Get an API key at [gpubridge.xyz](https://gpubridge.xyz) (free to start, pay per call).

## Usage

### Chat model

```python
from langchain_gpubridge import ChatGPUBridge

llm = ChatGPUBridge(api_key="gpub_...", service="llm-4090")
response = llm.invoke("What is GPU inference?")
print(response.content)
```

### Embeddings

```python
from langchain_gpubridge import GPUBridgeEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = GPUBridgeEmbeddings(api_key="gpub_...")
vectorstore = FAISS.from_texts(["text 1", "text 2"], embeddings)
docs = vectorstore.similarity_search("query")
```

### RAG pipeline

```python
from langchain_gpubridge import ChatGPUBridge, GPUBridgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

embeddings = GPUBridgeEmbeddings(api_key="gpub_...")
vectorstore = FAISS.from_texts(your_docs, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatGPUBridge(api_key="gpub_...", service="llm-4090")

prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n\n{context}\n\nQuestion: {question}"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

result = chain.invoke("Your question here")
print(result.content)
```

## Available services

| Service | Type | Description |
|---------|------|-------------|
| `llm-4090` | LLM | Llama 3.3 70B inference |
| `embedding-l4` | Embeddings | High-throughput text embeddings |
| `image-4090` | Image | FLUX / Stable Diffusion generation |
| `whisper-l4` | STT | Audio transcription |
| `tts-l4` | TTS | Text-to-speech |
| `rerank` | Reranking | Semantic reranking |
| `pdf-parse` | Documents | PDF text extraction |
| `nsfw-detect` | Moderation | Content moderation |

Full catalog: `curl https://api.gpubridge.xyz/catalog`

## x402 autonomous payments

GPU-Bridge supports [x402](https://x402.org) — agents can pay for inference with USDC on Base L2 without a pre-registered API key:

```python
# No API key needed — agent pays autonomously via x402
llm = ChatGPUBridge()  # will use x402 if wallet is configured
```

## License

MIT
