from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.entity_stores import BaseEntityStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
)
from langchain.memory.utils import get_prompt_input_key

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InMemoryEntityStore(BaseEntityStore):
    """In-memory Entity store."""

    store: Dict[str, Optional[str]] = {}

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.store.get(key, default)

    def set(self, key: str, value: Optional[str]) -> None:
        self.store[key] = value

    def delete(self, key: str) -> None:
        del self.store[key]

    def exists(self, key: str) -> bool:
        return key in self.store

    def clear(self) -> None:
        return self.store.clear()


class SQLiteEntityStore(BaseEntityStore):
    """SQLite-backed Entity store"""

    session_id: str = "default"
    table_name: str = "memory_store"
    conn: Any = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        session_id: str = "default",
        db_file: str = "entities.db",
        table_name: str = "memory_store",
        *args: Any,
        **kwargs: Any,
    ):
        try:
            import sqlite3
        except ImportError:
            raise ImportError(
                "Could not import sqlite3 python package. "
                "Please install it with `pip install sqlite3`."
            )
        super().__init__(*args, **kwargs)

        self.conn = sqlite3.connect(db_file)
        self.session_id = session_id
        self.table_name = table_name
        self._create_table_if_not_exists()

    @property
    def full_table_name(self) -> str:
        return f"{self.table_name}_{self.session_id}"

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        with self.conn:
            self.conn.execute(create_table_query)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        query = f"""
            SELECT value
            FROM {self.full_table_name}
            WHERE key = ?
        """
        cursor = self.conn.execute(query, (key,))
        result = cursor.fetchone()
        if result is not None:
            value = result[0]
            return value
        return default

    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        query = f"""
            INSERT OR REPLACE INTO {self.full_table_name} (key, value)
            VALUES (?, ?)
        """
        with self.conn:
            self.conn.execute(query, (key, value))

    def delete(self, key: str) -> None:
        query = f"""
            DELETE FROM {self.full_table_name}
            WHERE key = ?
        """
        with self.conn:
            self.conn.execute(query, (key,))

    def exists(self, key: str) -> bool:
        query = f"""
            SELECT 1
            FROM {self.full_table_name}
            WHERE key = ?
            LIMIT 1
        """
        cursor = self.conn.execute(query, (key,))
        result = cursor.fetchone()
        return result is not None

    def clear(self) -> None:
        query = f"""
            DELETE FROM {self.full_table_name}
        """
        with self.conn:
            self.conn.execute(query)


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
    entity_cache: List[str] = []

    # Number of recent message pairs to consider when updating entities:
    k: int = 3

    chat_history_key: str = "history"

    # Store to manage entity-related data:
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore)

    @property
    def buffer(self) -> List[BaseMessage]:
        """Access chat memory messages."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return ["entities", self.chat_history_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
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

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save context from this conversation history to the entity store.

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
