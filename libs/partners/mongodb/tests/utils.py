from __future__ import annotations

from copy import deepcopy
from time import monotonic, sleep
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Union, cast

from bson import ObjectId
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import model_validator
from pymongo.collection import Collection
from pymongo.results import DeleteResult, InsertManyResult

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.cache import MongoDBAtlasSemanticCache

TIMEOUT = 120
INTERVAL = 0.5


class PatchedMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearch):
    def bulk_embed_and_insert_texts(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List:
        """Patched insert_texts that waits for data to be indexed before returning"""
        ids_inserted = super().bulk_embed_and_insert_texts(texts, metadatas, ids)
        start = monotonic()
        while len(ids_inserted) != len(self.similarity_search("sandwich")) and (
            monotonic() - start <= TIMEOUT
        ):
            sleep(INTERVAL)
        return ids_inserted

    def create_vector_search_index(
        self,
        dimensions: int,
        filters: Optional[List[str]] = None,
        update: bool = False,
    ) -> None:
        result = super().create_vector_search_index(
            dimensions=dimensions, filters=filters, update=update
        )
        start = monotonic()
        while monotonic() - start <= TIMEOUT:
            if indexes := list(
                self._collection.list_search_indexes(name=self._index_name)
            ):
                if indexes[0].get("status") == "READY":
                    return result
            sleep(INTERVAL)


class ConsistentFakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [float(1.0)] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text))
            ]
            out_vectors.append(vector)
        return out_vectors

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class FakeChatModel(SimpleChatModel):
    """Fake Chat Model wrapper for testing purposes."""

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "fake response"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = "fake response"
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"key": "fake"}


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @model_validator(mode="before")
    @classmethod
    def check_queries_required(cls, values: dict) -> dict:
        if values.get("sequential_response") and not values.get("queries"):
            raise ValueError(
                "queries is required when sequential_response is set to True"
            )
        return values

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response


class MockCollection(Collection):
    """Mocked Mongo Collection"""

    _aggregate_result: List[Any]
    _insert_result: Optional[InsertManyResult]
    _data: List[Any]
    _simulate_cache_aggregation_query: bool

    def __init__(self) -> None:
        self._data = []
        self._aggregate_result = []
        self._insert_result = None
        self._simulate_cache_aggregation_query = False

    def delete_many(self, *args, **kwargs) -> DeleteResult:  # type: ignore
        old_len = len(self._data)
        self._data = []
        return DeleteResult({"n": old_len}, acknowledged=True)

    def insert_many(self, to_insert: List[Any], *args, **kwargs) -> InsertManyResult:  # type: ignore
        mongodb_inserts = [
            {"_id": ObjectId(), "score": 1, **insert} for insert in to_insert
        ]
        self._data.extend(mongodb_inserts)
        return self._insert_result or InsertManyResult(
            [k["_id"] for k in mongodb_inserts], acknowledged=True
        )

    def insert_one(self, to_insert: Any, *args, **kwargs) -> Any:  # type: ignore
        return self.insert_many([to_insert])

    def find_one(self, find_query: Dict[str, Any]) -> Optional[Dict[str, Any]]:  # type: ignore
        find = self.find(find_query) or [None]  # type: ignore
        return find[0]

    def find(self, find_query: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:  # type: ignore
        def _is_match(item: Dict[str, Any]) -> bool:
            for key, match_val in find_query.items():
                if item.get(key) != match_val:
                    return False
            return True

        return [document for document in self._data if _is_match(document)]

    def update_one(  # type: ignore
        self,
        find_query: Dict[str, Any],
        options: Dict[str, Any],
        *args: Any,
        upsert=True,
        **kwargs: Any,
    ) -> None:  # type: ignore
        result = self.find_one(find_query)
        set_options = options.get("$set", {})

        if result:
            result.update(set_options)
        elif upsert:
            self._data.append({**find_query, **set_options})

    def _execute_cache_aggregation_query(self, *args, **kwargs) -> List[Dict[str, Any]]:  # type: ignore
        """Helper function only to be used for MongoDBAtlasSemanticCache Testing

        Returns:
            List[Dict[str, Any]]: Aggregation query result
        """
        pipeline: List[Dict[str, Any]] = args[0]
        params = pipeline[0]["$vectorSearch"]
        embedding = params["queryVector"]
        # Assumes MongoDBAtlasSemanticCache.LLM == "llm_string"
        llm_string = params["filter"][MongoDBAtlasSemanticCache.LLM]["$eq"]

        acc = []
        for document in self._data:
            if (
                document.get("embedding") == embedding
                and document.get(MongoDBAtlasSemanticCache.LLM) == llm_string
            ):
                acc.append(document)
        return acc

    def aggregate(self, *args, **kwargs) -> List[Any]:  # type: ignore
        if self._simulate_cache_aggregation_query:
            return deepcopy(self._execute_cache_aggregation_query(*args, **kwargs))
        return deepcopy(self._aggregate_result)

    def count_documents(self, *args, **kwargs) -> int:  # type: ignore
        return len(self._data)

    def __repr__(self) -> str:
        return "MockCollection"
