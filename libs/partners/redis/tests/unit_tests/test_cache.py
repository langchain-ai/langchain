import json  # noqa: I001
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation
from langchain_redis import RedisCache, RedisSemanticCache
from unittest.mock import Mock, patch, MagicMock


class MockRedisJSON:
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}

    def set(self, key: str, path: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str) -> Any:
        return self.data.get(key)


class MockRedis:
    def __init__(self) -> None:
        self._json = MockRedisJSON()

    def json(self) -> MockRedisJSON:
        return self._json

    def expire(self, key: str, ttl: int) -> None:
        pass  # We're not implementing TTL in the mock

    def scan(
        self, cursor: int, match: Optional[str] = None, count: Optional[int] = None
    ) -> tuple[int, List[str]]:
        matching_keys = [
            k for k in self._json.data.keys() if match is None or k.startswith(match)
        ]
        return 0, matching_keys

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._json.data.pop(key, None)


# Helper functions (make sure these match the ones in your actual implementation)
def _serialize_generations(generations: List[Generation]) -> str:
    return json.dumps([gen.dict() for gen in generations])


def _deserialize_generations(generations_str: str) -> Optional[List[Generation]]:
    try:
        return [Generation(**gen) for gen in json.loads(generations_str)]
    except (json.JSONDecodeError, TypeError):
        return None


class MockRedisVLSemanticCache:
    def __init__(self) -> None:
        self.data: Dict[tuple, List[Dict[str, Any]]] = {}
        self.distance_threshold: float = 0.2  # Default value

    def check(self, vector: List[float]) -> List[Dict[str, Any]]:
        for stored_vector, stored_data in self.data.items():
            distance = np.linalg.norm(np.array(vector) - np.array(stored_vector))
            if distance <= self.distance_threshold:
                return stored_data
        return []

    def store(
        self,
        prompt: str,
        response: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data[tuple(vector)] = [{"response": response, "metadata": metadata}]

    def clear(self) -> None:
        self.data.clear()

    def _vectorize_prompt(self, prompt: str) -> List[float]:
        # Simple mock implementation, returns different vectors for different prompts
        return [hash(prompt) % 10 * 0.1, hash(prompt) % 7 * 0.1, hash(prompt) % 5 * 0.1]


class TestRedisCache:
    @pytest.fixture
    def redis_cache(self) -> RedisCache:
        mock_redis = Mock()
        mock_json = Mock()
        mock_redis.json.return_value = mock_json
        mock_json.set = Mock()
        mock_json.get = Mock(return_value=None)
        mock_redis.expire = MagicMock()
        mock_redis.scan = Mock(return_value=(0, []))
        mock_redis.delete = Mock()

        with patch("langchain_redis.cache.Redis.from_url", return_value=mock_redis):
            cache = RedisCache(redis_url="redis://localhost:6379", ttl=3600)
            cache.redis = mock_redis
            return cache

    def test_update_and_lookup(self, redis_cache: RedisCache) -> None:
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]

        # Mock data storage
        stored_data = {}

        def mock_set(key: str, path: str, value: Any) -> None:
            stored_data[key] = value

        def mock_get(key: str) -> Any:
            return stored_data.get(key)

        redis_cache.redis.json().set.side_effect = mock_set
        redis_cache.redis.json().get.side_effect = mock_get

        redis_cache.update(prompt, llm_string, return_val)
        result = redis_cache.lookup(prompt, llm_string)

        assert result is not None, "Lookup result should not be None"
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert (
            result[0].text == "test response"
        ), f"Expected 'test response', got '{result[0].text}'"

    def test_clear(self, redis_cache: RedisCache) -> None:
        prompt1, prompt2 = "test prompt 1", "test prompt 2"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_cache.update(prompt1, llm_string, return_val)
        redis_cache.update(prompt2, llm_string, return_val)

        redis_cache.clear()
        assert redis_cache.lookup(prompt1, llm_string) is None
        assert redis_cache.lookup(prompt2, llm_string) is None

    def test_ttl(self, redis_cache: RedisCache) -> None:
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]

        redis_cache.update(prompt, llm_string, return_val)

        key = redis_cache._key(prompt, llm_string)
        assert isinstance(redis_cache.redis.expire, MagicMock)
        redis_cache.redis.expire.assert_called_once_with(key, 3600)


class TestRedisSemanticCache:
    @pytest.fixture
    def mock_embeddings(self) -> Mock:
        embeddings = Mock(spec=Embeddings)
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        return embeddings

    @pytest.fixture
    def redis_semantic_cache(self, mock_embeddings: Mock) -> RedisSemanticCache:
        with patch(
            "langchain_redis.cache.RedisVLSemanticCache",
            return_value=MockRedisVLSemanticCache(),
        ):
            return RedisSemanticCache(
                embeddings=mock_embeddings, redis_url="redis://localhost:6379"
            )

    def test_update(self, redis_semantic_cache: RedisSemanticCache) -> None:
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt, llm_string, return_val)

        vector = redis_semantic_cache.cache._vectorize_prompt(prompt)
        assert redis_semantic_cache.cache.data[tuple(vector)] is not None

    def test_lookup(self, redis_semantic_cache: RedisSemanticCache) -> None:
        prompt = "test prompt"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt, llm_string, return_val)

        result = redis_semantic_cache.lookup(prompt, llm_string)

        assert result is not None
        assert len(result) == 1
        assert result[0].text == "test response"

        # Test lookup with different llm_string
        different_result = redis_semantic_cache.lookup(prompt, "different_llm")
        assert different_result is None

    def test_clear(self, redis_semantic_cache: RedisSemanticCache) -> None:
        prompt1, prompt2 = "test prompt 1", "test prompt 2"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt1, llm_string, return_val)
        redis_semantic_cache.update(prompt2, llm_string, return_val)

        redis_semantic_cache.clear()
        assert len(redis_semantic_cache.cache.data) == 0

    def test_distance_threshold(self, redis_semantic_cache: RedisSemanticCache) -> None:
        redis_semantic_cache.cache.distance_threshold = 0.1
        prompt1 = "test prompt 1"
        prompt2 = "test prompt 2"
        llm_string = "test_llm"
        return_val = [Generation(text="test response")]
        redis_semantic_cache.update(prompt1, llm_string, return_val)

        # Lookup with the same prompt should return the result
        result_same = redis_semantic_cache.lookup(prompt1, llm_string)
        assert result_same is not None
        assert len(result_same) == 1
        assert result_same[0].text == "test response"

        # Lookup with a different prompt should return None due to distance threshold
        result_different = redis_semantic_cache.lookup(prompt2, llm_string)
        assert result_different is None

        # Test with a higher distance threshold
        redis_semantic_cache.cache.distance_threshold = 1.0
        result_high_threshold = redis_semantic_cache.lookup(prompt2, llm_string)
        assert result_high_threshold is not None
        assert len(result_high_threshold) == 1
        assert result_high_threshold[0].text == "test response"
