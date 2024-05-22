import json
from typing import Any, Dict, Generator, List
from unittest.mock import Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition
from redis.commands.search.query import Query
from redis.commands.search.result import Result
from redis.exceptions import ResponseError

from langchain_redis import RedisChatMessageHistory


class MockRedisJSON:
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}

    def set(self, key: str, path: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def _load_data(self, key: str, data: Any) -> None:
        self.data[key] = data


class MockRedis:
    def __init__(self) -> None:
        self.indices: Dict[str, Any] = {}
        self._json = MockRedisJSON()

    def json(self) -> MockRedisJSON:
        return self._json

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._json.data.pop(key, None)

    def ft(self, index_name: str) -> "MockFT":
        if index_name not in self.indices:
            self.indices[index_name] = MockFT(self, index_name)
        return self.indices[index_name]

    def keys(self, pattern: str) -> List[str]:
        return [key for key in self._json.data.keys() if key.startswith(pattern)]

    def execute_command(self, *args: Any, **kwargs: Any) -> List[Any]:
        if args[0] == "FT.SEARCH":
            results = [
                ["id", key, "json", json.dumps(value)]
                for key, value in self._json.data.items()
            ]
            return [len(results), *results]
        else:
            return []


class MockFT:
    def __init__(self, redis: MockRedis, index_name: str) -> None:
        self.redis = redis
        self.index_name = index_name

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        pass

    def info(self) -> Dict[str, str]:
        return {"index_name": self.index_name}

    def search(self, query: Any) -> Mock:
        results = Mock()
        results.docs = [
            Mock(id=k, json=json.dumps(v)) for k, v in self.redis._json.data.items()
        ]
        results.total = len(results.docs)
        return results


class TestRedisChatMessageHistory:
    @pytest.fixture
    def mock_redis(self) -> MockRedis:
        return MockRedis()

    @pytest.fixture
    def chat_history(
        self, mock_redis: MockRedis
    ) -> Generator[RedisChatMessageHistory, None, None]:
        with patch("redis.Redis.from_url", return_value=mock_redis):
            history = RedisChatMessageHistory(
                session_id="test_session", redis_url="redis://localhost:6379"
            )
            yield history

    def test_initialization(self, chat_history: RedisChatMessageHistory) -> None:
        assert isinstance(chat_history, RedisChatMessageHistory)
        assert chat_history.session_id == "test_session"

    def test_add_message(self, chat_history: RedisChatMessageHistory) -> None:
        human_message = HumanMessage(content="Hello, AI!")
        ai_message = AIMessage(content="Hello, human!")
        system_message = SystemMessage(content="System message")

        chat_history.add_message(human_message)
        chat_history.add_message(ai_message)
        chat_history.add_message(system_message)

        messages = chat_history.messages
        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], SystemMessage)
        assert messages[0].content == "Hello, AI!"
        assert messages[1].content == "Hello, human!"
        assert messages[2].content == "System message"

    def test_clear(self, chat_history: RedisChatMessageHistory) -> None:
        chat_history.add_message(HumanMessage(content="Test message"))
        assert len(chat_history.messages) == 1

        chat_history.clear()
        assert len(chat_history.messages) == 0

    def test_search_messages(self, chat_history: RedisChatMessageHistory) -> None:
        with patch.object(chat_history.redis_client, "ft") as mock_ft:
            mock_search = Mock()
            mock_ft.return_value.search = mock_search

            # Create a mock search result
            mock_doc = Mock()
            mock_doc.id = "test_key"
            mock_doc.json = json.dumps(
                {
                    "data": {
                        "content": "What's the weather like today?",
                        "additional_kwargs": {},
                        "type": "human",
                    }
                }
            )

            mock_result = Mock(spec=Result)
            mock_result.docs = [mock_doc]
            mock_result.total = 1

            mock_search.return_value = mock_result

            results = chat_history.search_messages("weather", limit=5)

            # Check that ft was called with the correct index name
            mock_ft.assert_called_once_with(chat_history.index_name)

            # Check that search was called with the correct query
            mock_search.assert_called_once()
            query_arg = mock_search.call_args[0][0]

            assert isinstance(query_arg, Query)

            # Check the query string
            expected_query_string = (
                f"(@session_id:{{{chat_history.session_id}}}) (@content:weather)"
            )
            assert query_arg.query_string() == expected_query_string

            # Check the additional arguments
            args = query_arg.get_args()
            assert "(@session_id:{test_session}) (@content:weather)" in args
            assert "SORTBY" in args
            assert "timestamp" in args
            assert "ASC" in args
            assert "LIMIT" in args
            assert 0 in args
            assert 5 in args

            # Check the returned results
            assert len(results) == 1
            assert results[0]["content"] == "What's the weather like today?"
            assert results[0]["type"] == "human"

    def test_len(self, chat_history: RedisChatMessageHistory) -> None:
        chat_history.add_message(HumanMessage(content="Message 1"))
        chat_history.add_message(AIMessage(content="Message 2"))
        chat_history.add_message(HumanMessage(content="Message 3"))

        assert len(chat_history) == 3

    def test_ttl(self, chat_history: RedisChatMessageHistory) -> None:
        chat_history.add_message(HumanMessage(content="Test message"))
        assert len(chat_history.messages) == 1

        # Manually clear the data to simulate TTL expiration
        chat_history.redis_client._json.data.clear()  # type: ignore[attr-defined]
        assert len(chat_history.messages) == 0

    def test_ensure_index(self, chat_history: RedisChatMessageHistory) -> None:
        with patch.object(chat_history.redis_client, "ft") as mock_ft:
            # Mock the info method to raise the specific ResponseError
            mock_ft.return_value.info.side_effect = ResponseError("Unknown index name")

            # Call _ensure_index explicitly
            chat_history._ensure_index()

            # Check the calls made to ft
            assert mock_ft.call_count == 2
            assert mock_ft.call_args_list == [
                call(chat_history.index_name),
                call(chat_history.index_name),
            ]

            # Check info was called
            mock_ft.return_value.info.assert_called_once()

            # Check create_index was called and verify its arguments
            mock_ft.return_value.create_index.assert_called_once()
            create_index_args = mock_ft.return_value.create_index.call_args

            # Check the schema
            schema = create_index_args[0][0]
            assert len(schema) == 4

            # Check each field in the schema
            assert isinstance(schema[0], TagField)
            assert schema[0].redis_args() == [
                "$.session_id",
                "AS",
                "session_id",
                "TAG",
                "SEPARATOR",
                ",",
            ]

            assert isinstance(schema[1], TextField)
            assert schema[1].redis_args() == [
                "$.data.content",
                "AS",
                "content",
                "TEXT",
                "WEIGHT",
                1.0,
            ]

            assert isinstance(schema[2], TagField)
            assert schema[2].redis_args() == [
                "$.type",
                "AS",
                "type",
                "TAG",
                "SEPARATOR",
                ",",
            ]

            assert isinstance(schema[3], NumericField)
            assert schema[3].redis_args() == [
                "$.timestamp",
                "AS",
                "timestamp",
                "NUMERIC",
            ]

            # Check the index definition in the keyword arguments
            index_definition = create_index_args[1].get("definition")
            assert isinstance(index_definition, IndexDefinition)

            # Check the index definition args
            assert "ON" in index_definition.args
            assert "JSON" in index_definition.args
            assert "PREFIX" in index_definition.args
            assert 1 in index_definition.args  # The number of prefixes
            assert chat_history.key_prefix in index_definition.args

    def test_add_user_message(self, chat_history: RedisChatMessageHistory) -> None:
        chat_history.add_user_message("Hello, AI!")
        messages = chat_history.messages
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello, AI!"

    def test_add_ai_message(self, chat_history: RedisChatMessageHistory) -> None:
        chat_history.add_ai_message("Hello, human!")
        messages = chat_history.messages
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "Hello, human!"

    def test_multiple_sessions(self, mock_redis: MockRedis) -> None:
        # First session
        with patch("redis.Redis.from_url", return_value=mock_redis):
            history1 = RedisChatMessageHistory(
                session_id="session1", redis_url="redis://localhost:6379"
            )
            history1.add_user_message("Hello, AI!")
            history1.add_ai_message("Hello, how can I help you?")
            history1.add_user_message("Tell me a joke.")
            history1.add_ai_message(
                "Why did the chicken cross the road? To get to the other side!"
            )

        # Ensure the messages are added correctly in the first session
        messages1 = history1.messages
        assert len(messages1) == 4
        assert messages1[0].content == "Hello, AI!"
        assert messages1[1].content == "Hello, how can I help you?"
        assert messages1[2].content == "Tell me a joke."
        assert (
            messages1[3].content
            == "Why did the chicken cross the road? To get to the other side!"
        )

        # Second session
        with patch("redis.Redis.from_url", return_value=mock_redis):
            history2 = RedisChatMessageHistory(
                session_id="session1", redis_url="redis://localhost:6379"
            )

        # Ensure the history is maintained in the second session
        messages2 = history2.messages
        assert len(messages2) == 4
        assert messages2[0].content == "Hello, AI!"
        assert messages2[1].content == "Hello, how can I help you?"
        assert messages2[2].content == "Tell me a joke."
        assert (
            messages2[3].content
            == "Why did the chicken cross the road? To get to the other side!"
        )

    def test_add_user_and_ai_messages(
        self, chat_history: RedisChatMessageHistory
    ) -> None:
        chat_history.add_user_message("Hello!")
        chat_history.add_ai_message("Hi there!")

        messages = chat_history.messages
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Hello!"
        assert messages[1].content == "Hi there!"
