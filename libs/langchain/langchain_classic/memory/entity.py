"""Deprecated as of LangChain v0.3.4 and will be removed in LangChain v1.0.0."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING, Any

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_classic.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
)
from langchain_classic.memory.utils import get_prompt_input_key

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class BaseEntityStore(BaseModel, ABC):
    """Abstract base class for Entity store."""

    @abstractmethod
    def get(self, key: str, default: str | None = None) -> str | None:
        """Get entity value from store."""

    @abstractmethod
    def set(self, key: str, value: str | None) -> None:
        """Set entity value in store."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete entity value from store."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if entity exists in store."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all entities from store."""


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class InMemoryEntityStore(BaseEntityStore):
    """In-memory Entity store."""

    store: dict[str, str | None] = {}

    @override
    def get(self, key: str, default: str | None = None) -> str | None:
        return self.store.get(key, default)

    @override
    def set(self, key: str, value: str | None) -> None:
        self.store[key] = value

    @override
    def delete(self, key: str) -> None:
        del self.store[key]

    @override
    def exists(self, key: str) -> bool:
        return key in self.store

    @override
    def clear(self) -> None:
        return self.store.clear()


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class UpstashRedisEntityStore(BaseEntityStore):
    """Upstash Redis backed Entity store.

    Entities get a TTL of 1 day by default, and
    that TTL is extended by 3 days every time the entity is read back.
    """

    def __init__(
        self,
        session_id: str = "default",
        url: str = "",
        token: str = "",
        key_prefix: str = "memory_store",
        ttl: int | None = 60 * 60 * 24,
        recall_ttl: int | None = 60 * 60 * 24 * 3,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the RedisEntityStore.

        Args:
            session_id: Unique identifier for the session.
            url: URL of the Redis server.
            token: Authentication token for the Redis server.
            key_prefix: Prefix for keys in the Redis store.
            ttl: Time-to-live for keys in seconds (default 1 day).
            recall_ttl: Time-to-live extension for keys when recalled (default 3 days).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        try:
            from upstash_redis import Redis
        except ImportError as e:
            msg = (
                "Could not import upstash_redis python package. "
                "Please install it with `pip install upstash_redis`."
            )
            raise ImportError(msg) from e

        super().__init__(*args, **kwargs)

        try:
            self.redis_client = Redis(url=url, token=token)
        except Exception as exc:
            error_msg = "Upstash Redis instance could not be initiated"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from exc

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.recall_ttl = recall_ttl or ttl

    @property
    def full_key_prefix(self) -> str:
        """Returns the full key prefix with session ID."""
        return f"{self.key_prefix}:{self.session_id}"

    @override
    def get(self, key: str, default: str | None = None) -> str | None:
        res = (
            self.redis_client.getex(f"{self.full_key_prefix}:{key}", ex=self.recall_ttl)
            or default
            or ""
        )
        logger.debug(
            "Upstash Redis MEM get '%s:%s': '%s'", self.full_key_prefix, key, res
        )
        return res

    @override
    def set(self, key: str, value: str | None) -> None:
        if not value:
            return self.delete(key)
        self.redis_client.set(f"{self.full_key_prefix}:{key}", value, ex=self.ttl)
        logger.debug(
            "Redis MEM set '%s:%s': '%s' EX %s",
            self.full_key_prefix,
            key,
            value,
            self.ttl,
        )
        return None

    @override
    def delete(self, key: str) -> None:
        self.redis_client.delete(f"{self.full_key_prefix}:{key}")

    @override
    def exists(self, key: str) -> bool:
        return self.redis_client.exists(f"{self.full_key_prefix}:{key}") == 1

    @override
    def clear(self) -> None:
        def scan_and_delete(cursor: int) -> int:
            cursor, keys_to_delete = self.redis_client.scan(
                cursor,
                f"{self.full_key_prefix}:*",
            )
            self.redis_client.delete(*keys_to_delete)
            return cursor

        cursor = scan_and_delete(0)
        while cursor != 0:
            scan_and_delete(cursor)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class RedisEntityStore(BaseEntityStore):
    """Redis-backed Entity store.

    Entities get a TTL of 1 day by default, and
    that TTL is extended by 3 days every time the entity is read back.
    """

    redis_client: Any
    session_id: str = "default"
    key_prefix: str = "memory_store"
    ttl: int | None = 60 * 60 * 24
    recall_ttl: int | None = 60 * 60 * 24 * 3

    def __init__(
        self,
        session_id: str = "default",
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "memory_store",
        ttl: int | None = 60 * 60 * 24,
        recall_ttl: int | None = 60 * 60 * 24 * 3,
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the RedisEntityStore.

        Args:
            session_id: Unique identifier for the session.
            url: URL of the Redis server.
            key_prefix: Prefix for keys in the Redis store.
            ttl: Time-to-live for keys in seconds (default 1 day).
            recall_ttl: Time-to-live extension for keys when recalled (default 3 days).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        try:
            import redis
        except ImportError as e:
            msg = (
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
            raise ImportError(msg) from e

        super().__init__(*args, **kwargs)

        try:
            from langchain_community.utilities.redis import get_client
        except ImportError as e:
            msg = (
                "Could not import langchain_community.utilities.redis.get_client. "
                "Please install it with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        try:
            self.redis_client = get_client(redis_url=url, decode_responses=True)
        except redis.exceptions.ConnectionError:
            logger.exception("Redis client could not connect")

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.recall_ttl = recall_ttl or ttl

    @property
    def full_key_prefix(self) -> str:
        """Returns the full key prefix with session ID."""
        return f"{self.key_prefix}:{self.session_id}"

    @override
    def get(self, key: str, default: str | None = None) -> str | None:
        res = (
            self.redis_client.getex(f"{self.full_key_prefix}:{key}", ex=self.recall_ttl)
            or default
            or ""
        )
        logger.debug("REDIS MEM get '%s:%s': '%s'", self.full_key_prefix, key, res)
        return res

    @override
    def set(self, key: str, value: str | None) -> None:
        if not value:
            return self.delete(key)
        self.redis_client.set(f"{self.full_key_prefix}:{key}", value, ex=self.ttl)
        logger.debug(
            "REDIS MEM set '%s:%s': '%s' EX %s",
            self.full_key_prefix,
            key,
            value,
            self.ttl,
        )
        return None

    @override
    def delete(self, key: str) -> None:
        self.redis_client.delete(f"{self.full_key_prefix}:{key}")

    @override
    def exists(self, key: str) -> bool:
        return self.redis_client.exists(f"{self.full_key_prefix}:{key}") == 1

    @override
    def clear(self) -> None:
        # iterate a list in batches of size batch_size
        def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[Any]:
            iterator = iter(iterable)
            while batch := list(islice(iterator, batch_size)):
                yield batch

        for keybatch in batched(
            self.redis_client.scan_iter(f"{self.full_key_prefix}:*"),
            500,
        ):
            self.redis_client.delete(*keybatch)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class SQLiteEntityStore(BaseEntityStore):
    """SQLite-backed Entity store with safe query construction."""

    session_id: str = "default"
    table_name: str = "memory_store"
    conn: Any = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(
        self,
        session_id: str = "default",
        db_file: str = "entities.db",
        table_name: str = "memory_store",
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the SQLiteEntityStore.

        Args:
            session_id: Unique identifier for the session.
            db_file: Path to the SQLite database file.
            table_name: Name of the table to store entities.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        try:
            import sqlite3
        except ImportError as e:
            msg = (
                "Could not import sqlite3 python package. "
                "Please install it with `pip install sqlite3`."
            )
            raise ImportError(msg) from e

        # Basic validation to prevent obviously malicious table/session names
        if not table_name.isidentifier() or not session_id.isidentifier():
            # Since we validate here, we can safely suppress the S608 bandit warning
            msg = "Table name and session ID must be valid Python identifiers."
            raise ValueError(msg)

        self.conn = sqlite3.connect(db_file)
        self.session_id = session_id
        self.table_name = table_name
        self._create_table_if_not_exists()

    @property
    def full_table_name(self) -> str:
        """Returns the full table name with session ID."""
        return f"{self.table_name}_{self.session_id}"

    def _execute_query(self, query: str, params: tuple = ()) -> "sqlite3.Cursor":
        """Executes a query with proper connection handling."""
        with self.conn:
            return self.conn.execute(query, params)

    def _create_table_if_not_exists(self) -> None:
        """Creates the entity table if it doesn't exist, using safe quoting."""
        # Use standard SQL double quotes for the table name identifier
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS "{self.full_table_name}" (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        self._execute_query(create_table_query)

    def get(self, key: str, default: str | None = None) -> str | None:
        """Retrieves a value, safely quoting the table name."""
        # `?` placeholder is used for the value to prevent SQL injection
        # Ignore S608 since we validate for malicious table/session names in `__init__`
        query = f'SELECT value FROM "{self.full_table_name}" WHERE key = ?'  # noqa: S608
        cursor = self._execute_query(query, (key,))
        result = cursor.fetchone()
        return result[0] if result is not None else default

    def set(self, key: str, value: str | None) -> None:
        """Inserts or replaces a value, safely quoting the table name."""
        if not value:
            return self.delete(key)
        # Ignore S608 since we validate for malicious table/session names in `__init__`
        query = (
            "INSERT OR REPLACE INTO "  # noqa: S608
            f'"{self.full_table_name}" (key, value) VALUES (?, ?)'
        )
        self._execute_query(query, (key, value))
        return None

    def delete(self, key: str) -> None:
        """Deletes a key-value pair, safely quoting the table name."""
        # Ignore S608 since we validate for malicious table/session names in `__init__`
        query = f'DELETE FROM "{self.full_table_name}" WHERE key = ?'  # noqa: S608
        self._execute_query(query, (key,))

    def exists(self, key: str) -> bool:
        """Checks for the existence of a key, safely quoting the table name."""
        # Ignore S608 since we validate for malicious table/session names in `__init__`
        query = f'SELECT 1 FROM "{self.full_table_name}" WHERE key = ? LIMIT 1'  # noqa: S608
        cursor = self._execute_query(query, (key,))
        return cursor.fetchone() is not None

    @override
    def clear(self) -> None:
        # Ignore S608 since we validate for malicious table/session names in `__init__`
        query = f"""
            DELETE FROM {self.full_table_name}
        """  # noqa: S608
        with self.conn:
            self.conn.execute(query)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationEntityMemory(BaseChatMemory):
    """Entity extractor & summarizer memory.

    Extracts named entities from the recent chat history and generates summaries.
    With a swappable entity store, persisting entities across conversations.
    Defaults to an in-memory entity store, and can be swapped out for a Redis,
    SQLite, or other entity store.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT

    # Cache of recently detected entity names, if any
    # It is updated when load_memory_variables is called:
    entity_cache: list[str] = []

    # Number of recent message pairs to consider when updating entities:
    k: int = 3

    chat_history_key: str = "history"

    # Store to manage entity-related data:
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore)

    @property
    def buffer(self) -> list[BaseMessage]:
        """Access chat memory messages."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables."""
        return ["entities", self.chat_history_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load memory variables.

        Returns chat history and all generated entities with summaries if available,
        and updates or clears the recent entity cache.

        New entity name can be found when calling this method, before the entity
        summaries are generated, so the entity cache values may be empty if no entity
        descriptions are generated yet.
        """
        # Create an LLMChain for predicting entity names from the recent chat history:
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        # Extract an arbitrary window of the last message pairs from
        # the chat history, where the hyperparameter k is the
        # number of message pairs:
        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        # Generates a comma-separated list of named entities,
        # e.g. "Jane, White House, UFO"
        # or "NONE" if no named entities are extracted:
        output = chain.predict(
            history=buffer_string,
            input=inputs[prompt_input_key],
        )

        # If no named entities are extracted, assigns an empty list.
        if output.strip() == "NONE":
            entities = []
        else:
            # Make a list of the extracted entities:
            entities = [w.strip() for w in output.split(",")]

        # Make a dictionary of entities with summary if exists:
        entity_summaries = {}

        for entity in entities:
            entity_summaries[entity] = self.entity_store.get(entity, "")

        # Replaces the entity name cache with the most recently discussed entities,
        # or if no entities were extracted, clears the cache:
        self.entity_cache = entities

        # Should we return as message objects or as a string?
        if self.return_messages:
            # Get last `k` pair of chat messages:
            buffer: Any = self.buffer[-self.k * 2 :]
        else:
            # Reuse the string we made earlier:
            buffer = buffer_string

        return {
            self.chat_history_key: buffer,
            "entities": entity_summaries,
        }

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this conversation history to the entity store.

        Generates a summary for each entity in the entity cache by prompting
        the model, and saves these summaries to the entity store.
        """
        super().save_context(inputs, outputs)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        # Extract an arbitrary window of the last message pairs from
        # the chat history, where the hyperparameter k is the
        # number of message pairs:
        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        input_data = inputs[prompt_input_key]

        # Create an LLMChain for predicting entity summarization from the context
        chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)

        # Generate new summaries for entities and save them in the entity store
        for entity in self.entity_cache:
            # Get existing summary if it exists
            existing_summary = self.entity_store.get(entity, "")
            output = chain.predict(
                summary=existing_summary,
                entity=entity,
                history=buffer_string,
                input=input_data,
            )
            # Save the updated summary to the entity store
            self.entity_store.set(entity, output.strip())

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
        self.entity_cache.clear()
        self.entity_store.clear()
