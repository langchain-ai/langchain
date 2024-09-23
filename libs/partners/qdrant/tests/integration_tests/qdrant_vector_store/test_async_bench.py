import asyncio
import logging
import random
import uuid
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, models

from langchain_community.embeddings import LocalAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from tests.integration_tests.common import ConsistentFakeSparseEmbeddings

location = "http://localhost:6333"
vector_name = ""
retrieval_mode = RetrievalMode.DENSE
sparse_vector_name = "my-sparse-vector"

bulk_embeds = LocalAIEmbeddings(
    openai_api_base="http://localhost:9090/v1",
    model="bert-cpp-minilm-v6",
    openai_api_key="foo",
)


@pytest.fixture(scope="function")
def qvs() -> QdrantVectorStore:
    collection_name = "test_coll"

    instance = QdrantVectorStore.construct_instance(
        bulk_embeds,  # ConsistentFakeEmbeddings(),
        client_options={"location": location},
        collection_name=collection_name,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        force_recreate=True,
    )
    yield instance
    instance.client.delete_collection(collection_name)


@pytest.fixture(scope="class")
def texts() -> List[str]:
    def generate_syllable():
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        return random.choice(consonants) + random.choice(vowels)

    def generate_word():
        num_syllables = random.randint(1, 3)
        word = "".join(generate_syllable() for _ in range(num_syllables))
        return word.capitalize() if random.choice([True, False]) else word

    return [
        " ".join(generate_word() for _ in range(random.randint(3, 10)))
        for _ in range(10000)
    ]


class AsyncQVS(QdrantVectorStore):
    @classmethod
    def construct_instance(
        cls: Type[QdrantVectorStore],
        embedding: Optional[Embeddings] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        client_options: Dict[str, Any] = {},
        collection_name: Optional[str] = None,
        distance: models.Distance = models.Distance.COSINE,
        content_payload_key: str = QdrantVectorStore.CONTENT_KEY,
        metadata_payload_key: str = QdrantVectorStore.METADATA_KEY,
        vector_name: str = QdrantVectorStore.VECTOR_NAME,
        sparse_vector_name: str = QdrantVectorStore.SPARSE_VECTOR_NAME,
        force_recreate: bool = False,
        collection_create_options: Dict[str, Any] = {},
        vector_params: Dict[str, Any] = {},
        sparse_vector_params: Dict[str, Any] = {},
        validate_embeddings: bool = True,
        validate_collection_config: bool = True,
    ) -> QdrantVectorStore:
        instance = super().construct_instance(
            embedding,
            retrieval_mode,
            sparse_embedding,
            client_options,
            collection_name,
            distance,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            sparse_vector_name,
            force_recreate,
            collection_create_options,
            vector_params,
            sparse_vector_params,
            validate_embeddings,
            validate_collection_config,
        )
        instance.async_client = AsyncQdrantClient(**client_options)
        return instance

    async def _abuild_vectors(
        self,
        texts: Iterable[str],
    ) -> List[models.VectorStruct]:
        if self.retrieval_mode == RetrievalMode.DENSE:
            batch_embeddings = await self.embeddings.aembed_documents(list(texts))
            return [
                {
                    self.vector_name: vector,
                }
                for vector in batch_embeddings
            ]
        else:
            raise ValueError(
                f"Unknown retrieval mode. {self.retrieval_mode} to build vectors."
            )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
        max_parallel: int = 10,
        **kwargs: Any,
    ) -> List[str]:
        async def handle_batch(batch_texts, batch_metadatas, batch_ids, sem, **kwargs):
            async with sem:
                vectors = await self._abuild_vectors(batch_texts)
                points = [
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                    for point_id, vector, payload in zip(
                        batch_ids,
                        vectors,
                        self._build_payloads(
                            batch_texts,
                            batch_metadatas,
                            self.content_payload_key,
                            self.metadata_payload_key,
                        ),
                    )
                ]
                await self.async_client.upsert(
                    collection_name=self.collection_name, points=points, **kwargs
                )  ## care abt rzt?
            return batch_ids

        added_ids = []
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        sem = asyncio.Semaphore(max_parallel)
        async with asyncio.TaskGroup() as tg:
            while batch_texts := list(islice(texts_iterator, batch_size)):
                batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
                batch_ids = list(islice(ids_iterator, batch_size))
                tg.create_task(
                    handle_batch(batch_texts, batch_metadatas, batch_ids, sem, **kwargs)
                )
                added_ids.extend(batch_ids)
        return added_ids


@pytest.fixture(scope="function")
def async_qvs() -> AsyncQVS:
    async_name = "test_async"
    instance = AsyncQVS.construct_instance(
        bulk_embeds,  # ConsistentFakeEmbeddings(),
        client_options={"location": location},
        collection_name=async_name,
        vector_name=vector_name,
        sparse_vector_name=sparse_vector_name,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
        force_recreate=True,
    )
    yield instance
    instance.client.delete_collection(async_name)


def test_qdrant_add_texts(
    qvs: QdrantVectorStore, texts: List[str]
) -> None:
    qvs.add_texts(texts)  # , batch_size=10)
    act_count = qvs.client.count(qvs.collection_name)
    assert act_count.count == len(texts)


async def test_qdrant_async_aadd_texts(
        async_qvs: AsyncQVS, texts: List[str]
) -> None:
    await async_qvs.aadd_texts(texts, max_parallel=40)  # , batch_size=10)
    act_count = await async_qvs.async_client.count(async_qvs.collection_name)
    assert act_count.count == len(texts)
