"""Simple FastAPI service for querying a vector store."""

import os
from fastapi import FastAPI, Depends, Header, HTTPException, status
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.llms import HuggingFaceHub
try:
    from langchain_ollama import OllamaLLM
except Exception:  # pragma: no cover - optional dependency
    OllamaLLM = None
from .fake_llm import FakeLLM

try:
    from langchain_elasticsearch import ElasticsearchStore
except Exception:  # pragma: no cover
    ElasticsearchStore = None

try:
    from langchain_community.vectorstores import Weaviate
except Exception:  # pragma: no cover
    Weaviate = None

app = FastAPI()


def verify_token(x_api_token: str | None = Header(None)) -> str | None:
    expected = os.getenv("API_TOKEN")
    if expected and x_api_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return x_api_token

db_dir = os.environ.get("PERSIST_DIR", "chroma_db")
embedding_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
if embedding_model == "fake":
    embedding = FakeEmbeddings(size=10)
else:
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)

store_type = os.environ.get("VECTOR_STORE_TYPE", "chroma")
if store_type == "chroma":
    store = Chroma(persist_directory=db_dir, embedding_function=embedding)
elif store_type == "elastic" and ElasticsearchStore is not None:
    store = ElasticsearchStore(
        es_url=os.getenv("ES_URL", "http://localhost:9200"),
        index_name=os.getenv("ES_INDEX", "langchain_index"),
        embedding=embedding,
    )
elif store_type == "weaviate" and Weaviate is not None:
    store = Weaviate(
        url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        index_name=os.getenv("WEAVIATE_INDEX", "LangChain")
    )
else:
    raise ValueError(f"Unsupported store type: {store_type}")

# RetrievalQA setup mirrors the pattern in LangChain docs
use_fake = os.environ.get("USE_FAKE_LLM", "false").lower() == "true"
use_ollama = os.environ.get("USE_OLLAMA", "false").lower() == "true"
if use_fake:
    llm = FakeLLM()
elif use_ollama and OllamaLLM is not None:
    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3"))
else:
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever())

class Query(BaseModel):
    question: str

@app.post("/query")
async def query(q: Query, _: str | None = Depends(verify_token)):
    result = qa_chain.invoke({"query": q.question})
    return {"answer": result}

