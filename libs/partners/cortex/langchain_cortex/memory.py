"""Cortex-backed memory helper for LangChain chains and agents.

This module provides ``CortexMemory``, a convenience wrapper around
``CortexChatMessageHistory`` that exposes the ``load_memory_variables`` /
``save_context`` / ``clear`` interface familiar from the classic
``ConversationBufferMemory``, while storing all data persistently inside a
local Cortex database.

The recommended modern pattern is to use ``CortexChatMessageHistory`` directly
with ``RunnableWithMessageHistory``.  ``CortexMemory`` is provided for teams
migrating from ``ConversationBufferMemory`` or ``ConversationSummaryMemory``.

Requires the ``cortex-ai-memory`` package::

    pip install cortex-ai-memory

Usage example::

    from langchain_cortex import CortexMemory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    memory = CortexMemory(
        db_path="~/.cortex/agent.db",
        session_id="agent-session-1",
        memory_key="history",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | ChatOpenAI()

    # First turn
    response = chain.invoke({"history": [], "input": "My name is Alvin."})
    memory.save_context(
        {"input": "My name is Alvin."},
        {"output": response.content},
    )

    # Second turn — history is loaded from Cortex
    vars_ = memory.load_memory_variables({})
    response2 = chain.invoke({**vars_, "input": "What is my name?"})

Recommended pattern with ``RunnableWithMessageHistory``::

    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_cortex import CortexChatMessageHistory

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: CortexChatMessageHistory(
            db_path="~/.cortex/chat.db",
            session_id=session_id,
        ),
        input_messages_key="input",
        history_messages_key="history",
    )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_cortex.chat_history import CortexChatMessageHistory

_DEFAULT_HUMAN_PREFIX = "Human"
_DEFAULT_AI_PREFIX = "AI"


class CortexMemory(BaseModel):
    """Convenience memory wrapper backed by a local Cortex database.

    Stores conversation history as an ordered sequence of ``HumanMessage`` /
    ``AIMessage`` pairs in a ``CortexChatMessageHistory`` instance, and exposes
    them to LangChain chains via ``load_memory_variables`` / ``save_context``.

    For new code, prefer using ``CortexChatMessageHistory`` directly with
    ``RunnableWithMessageHistory`` instead of this class.

    Parameters:
        db_path: Path to the Cortex SQLite database file (``~`` is expanded).
        session_id: Logical conversation identifier.  Multiple sessions can
            co-exist inside a single database.
        memory_key: The key under which chat history is injected into the
            chain's input dict.  Defaults to ``"history"``.
        input_key: Chain input variable that contains the human turn text.
            When ``None`` the first key in the chain input dict is used.
        output_key: Chain output variable that contains the AI response text.
            When ``None`` the first key in the chain output dict is used.
        return_messages: When ``True`` (default) ``load_memory_variables``
            returns a list of ``BaseMessage`` objects suitable for
            ``MessagesPlaceholder``.  When ``False`` the history is returned as
            a formatted string.
        human_prefix: Display prefix for human turns in string mode.
        ai_prefix: Display prefix for AI turns in string mode.
        channel: Cortex channel name.  Defaults to ``"chat"``.
        salience: Base salience score assigned to stored messages.

    Example:
        .. code-block:: python

            from langchain_cortex import CortexMemory

            memory = CortexMemory(
                db_path="~/.cortex/memory.db",
                session_id="demo",
            )
            memory.save_context(
                {"input": "Hello"},
                {"output": "Hi there!"},
            )
            print(memory.load_memory_variables({}))
            # {'history': [HumanMessage(...), AIMessage(...)]}
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Declared fields — become constructor parameters automatically.
    db_path: str
    session_id: str
    memory_key: str = "history"
    input_key: str | None = None
    output_key: str | None = None
    return_messages: bool = True
    human_prefix: str = _DEFAULT_HUMAN_PREFIX
    ai_prefix: str = _DEFAULT_AI_PREFIX
    channel: str = "chat"
    salience: float = 0.6

    # Private — not serialised.  model_config allows arbitrary types.
    _chat_history: CortexChatMessageHistory | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _history(self) -> CortexChatMessageHistory:
        """Lazily initialise and cache the underlying chat history object."""
        if self._chat_history is None:
            self._chat_history = CortexChatMessageHistory(
                db_path=self.db_path,
                session_id=self.session_id,
                channel=self.channel,
                salience=self.salience,
            )
        return self._chat_history

    def _resolve_key(
        self, mapping: dict[str, Any], preferred: str | None
    ) -> str:
        if preferred is not None and preferred in mapping:
            return preferred
        keys = list(mapping.keys())
        if not keys:
            msg = "Cannot resolve key from empty mapping."
            raise ValueError(msg)
        return keys[0]

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    @property
    def memory_variables(self) -> list[str]:
        """Return the list of variable names this memory provides.

        Returns:
            A single-element list containing ``memory_key``.
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load conversation history from Cortex and return it for chain injection.

        Args:
            inputs: The chain's current input dict (unused by default; provided
                for API compatibility).

        Returns:
            A dict mapping ``memory_key`` to either a list of ``BaseMessage``
            objects (when ``return_messages=True``) or a formatted string.
        """
        messages: list[BaseMessage] = self._history.messages

        if self.return_messages:
            return {self.memory_key: messages}

        # String mode: interleave Human / AI prefixes.
        lines: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"{self.human_prefix}: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"{self.ai_prefix}: {msg.content}")
            else:
                lines.append(str(msg.content))
        return {self.memory_key: "\n".join(lines)}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> None:
        """Persist a single human/AI exchange to Cortex.

        Args:
            inputs: The chain's input dict; the human message is extracted
                using ``input_key`` (or the first key if unset).
            outputs: The chain's output dict; the AI response is extracted
                using ``output_key`` (or the first key if unset).
        """
        input_key = self._resolve_key(inputs, self.input_key)
        output_key = self._resolve_key(outputs, self.output_key)

        human_text = str(inputs[input_key])
        ai_text = str(outputs[output_key])

        self._history.add_messages(
            [HumanMessage(content=human_text), AIMessage(content=ai_text)]
        )

    def clear(self) -> None:
        """Clear all stored messages for this session.

        Delegates to ``CortexChatMessageHistory.clear()``.
        """
        self._history.clear()

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def get_context(self, max_tokens: int = 2000) -> str:
        """Return a token-budgeted context summary from Cortex.

        This surfaces Cortex's multi-tier context generation, which ranks
        memories by recency, salience, and semantic relevance rather than
        returning a raw chronological buffer.

        Args:
            max_tokens: Upper bound on context length in tokens.

        Returns:
            A formatted string suitable for insertion into a system prompt.
        """
        return self._history.get_context(max_tokens)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.9,
    ) -> str:
        """Store a semantic fact alongside the conversation history.

        Facts are stored in Cortex's semantic tier and survive
        consolidation, making them useful for long-term user profile data.

        Args:
            subject: The entity the fact describes (e.g. ``"User"``).
            predicate: The relationship (e.g. ``"works_at"``).
            obj: The value (e.g. ``"Acme Corp"``).
            confidence: Confidence score in [0, 1].

        Returns:
            The memory ID of the newly created fact record.
        """
        return self._history.cortex.add_fact(
            subject, predicate, obj, confidence, self.channel
        )

    def add_preference(self, key: str, value: str, confidence: float = 0.8) -> str:
        """Store a user preference in Cortex.

        Args:
            key: Preference key (e.g. ``"editor"``).
            value: Preference value (e.g. ``"neovim"``).
            confidence: Confidence score in [0, 1].

        Returns:
            The memory ID of the newly created preference record.
        """
        return self._history.cortex.add_preference(key, value, confidence)

    @property
    def cortex(self):  # type: ignore[return]  # noqa: ANN201
        """Direct access to the underlying ``PyCortex`` instance.

        Returns:
            The ``PyCortex`` instance shared by this memory object.
        """
        return self._history.cortex
