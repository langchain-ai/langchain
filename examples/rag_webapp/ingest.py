"""Example ingestion script supporting multiple vector stores.

Based on LangChain documentation for `ElasticsearchStore` and `Weaviate` in
`docs/docs/how_to/indexing.ipynb` and
`docs/docs/integrations/vectorstores/weaviate.ipynb`.
"""

import argparse
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
try:
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback for splitters package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma

try:  # Optional imports for other stores
    from langchain_elasticsearch import ElasticsearchStore
except Exception:  # pragma: no cover - dependency may be missing
    ElasticsearchStore = None

try:
    from langchain_community.vectorstores import Weaviate
except Exception:  # pragma: no cover
    Weaviate = None


def ingest(
    source_dir: str,
    persist_dir: str = "db",
    store_type: str = "chroma",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Ingest markdown docs into a selected vector store.

    Parameters are inspired by examples in LangChain docs. ``store_type`` can be
    ``chroma``, ``elastic`` or ``weaviate``. ``embedding_model`` selects the
    HuggingFace model or ``fake`` for testing.
    """
    loader = DirectoryLoader(
        source_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    doc_count = len(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if embedding_model == "fake":
        embeddings = FakeEmbeddings(size=10)
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )
    if store_type == "chroma":
        vectordb = Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
    elif store_type == "elastic" and ElasticsearchStore is not None:
        vectordb = ElasticsearchStore.from_documents(
            splits,
            embeddings,
            es_url=os.getenv("ES_URL", "http://localhost:9200"),
            index_name=os.getenv("ES_INDEX", "langchain_index"),
        )
    elif store_type == "weaviate" and Weaviate is not None:
        vectordb = Weaviate.from_documents(
            splits,
            embeddings,
            weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        )
    else:
        raise ValueError(f"Unsupported store type: {store_type}")
    vectordb.persist()
    print(
        f"Loaded {doc_count} files -> {len(splits)} chunks. Ingested into {store_type} store at {persist_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest markdown docs")
    parser.add_argument("--source", default=os.environ.get("SOURCE_DIR", "docs"))
    parser.add_argument("--persist", default=os.environ.get("PERSIST_DIR", "chroma_db"))
    parser.add_argument("--store", default=os.environ.get("VECTOR_STORE_TYPE", "chroma"))
    parser.add_argument(
        "--embedding",
        default=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    args = parser.parse_args()

    Path(args.persist).mkdir(parents=True, exist_ok=True)
    ingest(args.source, args.persist, args.store, args.embedding)


if __name__ == "__main__":
    main()
