from pathlib import Path
from typing import Any, Dict, Generator, Tuple

import numpy as np
import pytest
from docarray import BaseDoc
from docarray.index import (
    ElasticDocIndex,
    HnswDocumentIndex,
    InMemoryExactNNIndex,
    QdrantDocumentIndex,
    WeaviateDocumentIndex,
)
from docarray.typing import NdArray
from pydantic import Field
from qdrant_client.http import models as rest

from langchain.embeddings import FakeEmbeddings


class MyDoc(BaseDoc):
    title: str
    title_embedding: NdArray[32]  # type: ignore
    other_emb: NdArray[32]  # type: ignore
    year: int


class WeaviateDoc(BaseDoc):
    # When initializing the Weaviate index, denote the field
    # you want to search on with `is_embedding=True`
    title: str
    title_embedding: NdArray[32] = Field(is_embedding=True)  # type: ignore
    other_emb: NdArray[32]  # type: ignore
    year: int


@pytest.fixture
def init_weaviate() -> (
    Generator[
        Tuple[WeaviateDocumentIndex[WeaviateDoc], Dict[str, Any], FakeEmbeddings],
        None,
        None,
    ]
):
    """
    cd tests/integration_tests/vectorstores/docker-compose
    docker compose -f weaviate.yml up
    """
    embeddings = FakeEmbeddings(size=32)

    # initialize WeaviateDocumentIndex
    dbconfig = WeaviateDocumentIndex.DBConfig(host="http://localhost:8080")
    weaviate_db = WeaviateDocumentIndex[WeaviateDoc](
        db_config=dbconfig, index_name="docarray_retriever"
    )

    # index data
    weaviate_db.index(
        [
            WeaviateDoc(
                title=f"My document {i}",
                title_embedding=np.array(embeddings.embed_query(f"fake emb {i}")),
                other_emb=np.array(embeddings.embed_query(f"other fake emb {i}")),
                year=i,
            )
            for i in range(100)
        ]
    )
    # build a filter query
    filter_query = {"path": ["year"], "operator": "LessThanEqual", "valueInt": "90"}

    yield weaviate_db, filter_query, embeddings

    weaviate_db._client.schema.delete_all()


@pytest.fixture
def init_elastic() -> (
    Generator[Tuple[ElasticDocIndex[MyDoc], Dict[str, Any], FakeEmbeddings], None, None]
):
    """
    cd tests/integration_tests/vectorstores/docker-compose
    docker-compose -f elasticsearch.yml up
    """
    embeddings = FakeEmbeddings(size=32)

    # initialize ElasticDocIndex
    elastic_db = ElasticDocIndex[MyDoc](
        hosts="http://localhost:9200", index_name="docarray_retriever"
    )
    # index data
    elastic_db.index(
        [
            MyDoc(
                title=f"My document {i}",
                title_embedding=np.array(embeddings.embed_query(f"fake emb {i}")),
                other_emb=np.array(embeddings.embed_query(f"other fake emb {i}")),
                year=i,
            )
            for i in range(100)
        ]
    )
    # build a filter query
    filter_query = {"range": {"year": {"lte": 90}}}

    yield elastic_db, filter_query, embeddings

    elastic_db._client.indices.delete(index="docarray_retriever")


@pytest.fixture
def init_qdrant() -> Tuple[QdrantDocumentIndex[MyDoc], rest.Filter, FakeEmbeddings]:
    embeddings = FakeEmbeddings(size=32)

    # initialize QdrantDocumentIndex
    qdrant_config = QdrantDocumentIndex.DBConfig(path=":memory:")
    qdrant_db = QdrantDocumentIndex[MyDoc](qdrant_config)
    # index data
    qdrant_db.index(
        [
            MyDoc(
                title=f"My document {i}",
                title_embedding=np.array(embeddings.embed_query(f"fake emb {i}")),
                other_emb=np.array(embeddings.embed_query(f"other fake emb {i}")),
                year=i,
            )
            for i in range(100)
        ]
    )
    # build a filter query
    filter_query = rest.Filter(
        must=[
            rest.FieldCondition(
                key="year",
                range=rest.Range(
                    gte=10,
                    lt=90,
                ),
            )
        ]
    )

    return qdrant_db, filter_query, embeddings


@pytest.fixture
def init_in_memory() -> (
    Tuple[InMemoryExactNNIndex[MyDoc], Dict[str, Any], FakeEmbeddings]
):
    embeddings = FakeEmbeddings(size=32)

    # initialize InMemoryExactNNIndex
    in_memory_db = InMemoryExactNNIndex[MyDoc]()
    # index data
    in_memory_db.index(
        [
            MyDoc(
                title=f"My document {i}",
                title_embedding=np.array(embeddings.embed_query(f"fake emb {i}")),
                other_emb=np.array(embeddings.embed_query(f"other fake emb {i}")),
                year=i,
            )
            for i in range(100)
        ]
    )
    # build a filter query
    filter_query = {"year": {"$lte": 90}}

    return in_memory_db, filter_query, embeddings


@pytest.fixture
def init_hnsw(
    tmp_path: Path,
) -> Tuple[HnswDocumentIndex[MyDoc], Dict[str, Any], FakeEmbeddings]:
    embeddings = FakeEmbeddings(size=32)

    # initialize InMemoryExactNNIndex
    hnsw_db = HnswDocumentIndex[MyDoc](work_dir=tmp_path)
    # index data
    hnsw_db.index(
        [
            MyDoc(
                title=f"My document {i}",
                title_embedding=np.array(embeddings.embed_query(f"fake emb {i}")),
                other_emb=np.array(embeddings.embed_query(f"other fake emb {i}")),
                year=i,
            )
            for i in range(100)
        ]
    )
    # build a filter query
    filter_query = {"year": {"$lte": 90}}

    return hnsw_db, filter_query, embeddings
