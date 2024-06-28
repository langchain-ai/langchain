"""
LangChain Couchbase Caches

Functions "_hash", "_loads_generations" and "_dumps_generations"
are duplicated in this utility from modules:
    - "libs/community/langchain_community/cache.py"
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Union

from couchbase.cluster import Cluster
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

from langchain_couchbase.vectorstores import CouchbaseVectorStore

logger = logging.getLogger(__file__)


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


def _dumps_generations(generations: RETURN_VAL_TYPE) -> str:
    """
    Serialization for generic RETURN_VAL_TYPE, i.e. sequence of `Generation`

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: a single string representing a list of generations.

    This function (+ its counterpart `_loads_generations`) rely on
    the dumps/loads pair with Reviver, so are able to deal
    with all subclasses of Generation.

    Each item in the list can be `dumps`ed to a string,
    then we make the whole list of strings into a json-dumped.
    """
    return json.dumps([dumps(_item) for _item in generations])


def _loads_generations(generations_str: str) -> Union[RETURN_VAL_TYPE, None]:
    """
    Deserialization of a string into a generic RETURN_VAL_TYPE
    (i.e. a sequence of `Generation`).

    See `_dumps_generations`, the inverse of this function.

    Args:
        generations_str (str): A string representing a list of generations.

    Compatible with the legacy cache-blob format
    Does not raise exceptions for malformed entries, just logs a warning
    and returns none: the caller should be prepared for such a cache miss.

    Returns:
        RETURN_VAL_TYPE: A list of generations.
    """
    try:
        generations = [loads(_item_str) for _item_str in json.loads(generations_str)]
        return generations
    except (json.JSONDecodeError, TypeError):
        # deferring the (soft) handling to after the legacy-format attempt
        pass

    try:
        gen_dicts = json.loads(generations_str)
        # not relying on `_load_generations_from_json` (which could disappear):
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(
            f"Legacy 'Generation' cached blob encountered: '{generations_str}'"
        )
        return generations
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            f"Malformed/unparsable cached blob encountered: '{generations_str}'"
        )
        return None


class CouchbaseCache(BaseCache):
    """Couchbase LLM Cache
    LLM Cache that uses Couchbase as the backend
    """

    PROMPT = "prompt"
    LLM = "llm"
    RETURN_VAL = "return_val"

    def _check_bucket_exists(self) -> bool:
        """Check if the bucket exists in the linked Couchbase cluster"""
        bucket_manager = self._cluster.buckets()
        try:
            bucket_manager.get_bucket(self._bucket_name)
            return True
        except Exception:
            return False

    def _check_scope_and_collection_exists(self) -> bool:
        """Check if the scope and collection exists in the linked Couchbase bucket
        Raises a ValueError if either is not found"""
        scope_collection_map: Dict[str, Any] = {}

        # Get a list of all scopes in the bucket
        for scope in self._bucket.collections().get_all_scopes():
            scope_collection_map[scope.name] = []

            # Get a list of all the collections in the scope
            for collection in scope.collections:
                scope_collection_map[scope.name].append(collection.name)

        # Check if the scope exists
        if self._scope_name not in scope_collection_map.keys():
            raise ValueError(
                f"Scope {self._scope_name} not found in Couchbase "
                f"bucket {self._bucket_name}"
            )

        # Check if the collection exists in the scope
        if self._collection_name not in scope_collection_map[self._scope_name]:
            raise ValueError(
                f"Collection {self._collection_name} not found in scope "
                f"{self._scope_name} in Couchbase bucket {self._bucket_name}"
            )

        return True

    def __init__(
        self,
        cluster: Cluster,
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the Couchbase LLM Cache
        Args:
            cluster (Cluster): couchbase cluster object with active connection.
            bucket_name (str): name of the bucket to store documents in.
            scope_name (str): name of the scope in bucket to store documents in.
            collection_name (str): name of the collection in the scope to store
                documents in.
        """
        if not isinstance(cluster, Cluster):
            raise ValueError(
                f"cluster should be an instance of couchbase.Cluster, "
                f"got {type(cluster)}"
            )

        self._cluster = cluster

        self._bucket_name = bucket_name
        self._scope_name = scope_name
        self._collection_name = collection_name

        # Check if the bucket exists
        if not self._check_bucket_exists():
            raise ValueError(
                f"Bucket {self._bucket_name} does not exist. "
                " Please create the bucket before searching."
            )

        try:
            self._bucket = self._cluster.bucket(self._bucket_name)
            self._scope = self._bucket.scope(self._scope_name)
            self._collection = self._scope.collection(self._collection_name)
        except Exception as e:
            raise ValueError(
                "Error connecting to couchbase. "
                "Please check the connection and credentials."
            ) from e

        # Check if the scope and collection exists. Throws ValueError if they don't
        try:
            self._check_scope_and_collection_exists()
        except Exception as e:
            raise e

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up from cache based on prompt and llm_string."""
        try:
            doc = self._collection.get(
                self._generate_key(prompt, llm_string)
            ).content_as[dict]
            return _loads_generations(doc[self.RETURN_VAL])
        except Exception:
            return None

    def _generate_key(self, prompt: str, llm_string: str) -> str:
        """Generate the key based on prompt and llm_string."""
        return _hash(prompt + llm_string)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        doc = {
            self.PROMPT: prompt,
            self.LLM: llm_string,
            self.RETURN_VAL: _dumps_generations(return_val),
        }
        try:
            self._collection.upsert(
                key=self._generate_key(prompt, llm_string), value=doc
            )
        except Exception:
            logger.error("Error updating cache")

    def clear(self, **kwargs: Any) -> None:
        """Clear the cache.
        This will delete all documents in the collection. This requires an index on the
        collection.
        """
        try:
            query = f"DELETE FROM `{self._collection_name}`"
            self._scope.query(query).execute()
        except Exception:
            logger.error("Error clearing cache. Please check if you have an index.")


class CouchbaseSemanticCache(BaseCache, CouchbaseVectorStore):
    """Couchbase Semantic Cache
    Cache backed by a Couchbase Server with Vector Store support
    """

    LLM = "llm_string"
    RETURN_VAL = "return_val"

    def __init__(
        self,
        cluster: Cluster,
        embedding: Embeddings,
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        index_name: str,
        score_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the Couchbase LLM Cache
        Args:
            cluster (Cluster): couchbase cluster object with active connection.
            embedding (Embeddings): embedding model to use.
            bucket_name (str): name of the bucket to store documents in.
            scope_name (str): name of the scope in bucket to store documents in.
            collection_name (str): name of the collection in the scope to store
                documents in.
            index_name (str): name of the Search index to use.
            score_threshold (float): score threshold to use for filtering results.
        """
        if not isinstance(cluster, Cluster):
            raise ValueError(
                f"cluster should be an instance of couchbase.Cluster, "
                f"got {type(cluster)}"
            )

        self._cluster = cluster

        self._bucket_name = bucket_name
        self._scope_name = scope_name
        self._collection_name = collection_name

        # Check if the bucket exists
        if not self._check_bucket_exists():
            raise ValueError(
                f"Bucket {self._bucket_name} does not exist. "
                " Please create the bucket before searching."
            )

        try:
            self._bucket = self._cluster.bucket(self._bucket_name)
            self._scope = self._bucket.scope(self._scope_name)
            self._collection = self._scope.collection(self._collection_name)
        except Exception as e:
            raise ValueError(
                "Error connecting to couchbase. "
                "Please check the connection and credentials."
            ) from e

        # Check if the scope and collection exists. Throws ValueError if they don't
        try:
            self._check_scope_and_collection_exists()
        except Exception as e:
            raise e

        self.score_threshold = score_threshold

        # Initialize the vector store
        super().__init__(
            cluster=cluster,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            embedding=embedding,
            index_name=index_name,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up from cache based on the semantic similarity of the prompt"""
        search_results = self.similarity_search_with_score(
            prompt, k=1, search_options={f"metadata.{self.LLM}": llm_string}
        )
        if search_results:
            selected_doc, score = search_results[0]
        else:
            return None

        # Check if the score is above the threshold if a threshold is provided
        if self.score_threshold:
            if score < self.score_threshold:
                return None

        # Note that the llm_string might not match the vector search result.
        # So if the llm_string does not match, do not return the result.
        if selected_doc.metadata["llm_string"] != llm_string:
            return None

        return _loads_generations(selected_doc.metadata[self.RETURN_VAL])

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on the prompt and llm_string"""
        try:
            self.add_texts(
                texts=[prompt],
                metadatas=[
                    {
                        self.LLM: llm_string,
                        self.RETURN_VAL: _dumps_generations(return_val),
                    }
                ],
            )
        except Exception:
            logger.error("Error updating cache")

    def clear(self, **kwargs: Any) -> None:
        """Clear the cache.
        This will delete all documents in the collection.
        This requires an index on the collection.
        """
        try:
            query = f"DELETE FROM `{self._collection_name}`"
            self._scope.query(query).execute()
        except Exception:
            logger.error("Error clearing cache. Please check if you have an index.")
