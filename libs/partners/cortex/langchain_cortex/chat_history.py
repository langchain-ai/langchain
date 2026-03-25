"""Cortex-backed chat message history for LangChain.

This module provides ``CortexChatMessageHistory``, a ``BaseChatMessageHistory``
implementation that persists conversation turns inside a local Cortex database.

Each message is stored as a plain-text memory in the ``chat`` channel.  On
retrieval the full ordered history is reconstructed from the database using a
``session_id`` prefix so that multiple independent conversations can share a
single Cortex file without collision.

Requires the ``cortex-ai-memory`` package::

    pip install cortex-ai-memory

Usage example::

    from langchain_cortex import CortexChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    history = CortexChatMessageHistory(
        db_path="~/.cortex/chat.db",
        session_id="my-session",
    )

    history.add_user_message("Hello!")
    history.add_ai_message("Hi there, how can I help?")

    for msg in history.messages:
        print(msg)

    history.clear()

The history object can be plugged directly into
``RunnableWithMessageHistory``::

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

import json
import os
from typing import TYPE_CHECKING, Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

_CHANNEL = "chat"
# Prefix used when ingesting messages so we can filter by session.
_SESSION_PREFIX = "__session__"
# Maximum number of memories to retrieve when loading history.
_RETRIEVE_LIMIT = 500


def _open_cortex(db_path: str) -> Any:
    """Open a Cortex database and return the ``PyCortex`` instance.

    This thin wrapper is the single place where ``cortex_python`` is imported,
    making it straightforward to patch in tests without needing ``cortex-ai-memory``
    installed.

    Args:
        db_path: Filesystem path to the Cortex SQLite file.

    Returns:
        A ``PyCortex`` instance connected to ``db_path``.

    Raises:
        ImportError: If ``cortex-ai-memory`` is not installed.
    """
    try:
        from cortex_python import PyCortex  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = (
            "Could not import cortex-ai-memory. "
            "Install it with: pip install cortex-ai-memory"
        )
        raise ImportError(msg) from exc

    return PyCortex(db_path)


class CortexChatMessageHistory(BaseChatMessageHistory):
    """Persist LangChain chat history inside a local Cortex memory database.

    Parameters:
        db_path: Path to the Cortex SQLite database file (``~`` is expanded).
            The file is created automatically if it does not exist.
        session_id: Logical identifier for this conversation. Multiple sessions
            can share the same database file.
        channel: Cortex channel name used when ingesting messages.  Defaults to
            ``"chat"``.  Override this to namespace messages alongside other
            agent channels stored in the same database.
        salience: Base salience score (0–1) assigned to each stored message.
            Higher values are retrieved more eagerly by the Cortex retriever.
            Defaults to ``0.6``.

    Attributes:
        messages: The ordered list of messages for this session.  Each access
            round-trips to Cortex, so cache the result locally if you need it
            multiple times in a hot path.

    Example:
        .. code-block:: python

            from langchain_cortex import CortexChatMessageHistory

            history = CortexChatMessageHistory(
                db_path="~/.cortex/chat.db",
                session_id="user-42",
            )
            history.add_user_message("What is the capital of France?")
    """

    def __init__(
        self,
        db_path: str,
        session_id: str,
        *,
        channel: str = _CHANNEL,
        salience: float = 0.6,
    ) -> None:
        self._session_id = session_id
        self._channel = channel
        self._salience = salience
        self._cx = _open_cortex(os.path.expanduser(db_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_tag(self) -> str:
        """Return a deterministic prefix that namespaces this session's messages."""
        return f"{_SESSION_PREFIX}{self._session_id}::"

    def _pack(self, message: BaseMessage) -> str:
        """Serialize a single message to a JSON string for storage."""
        return json.dumps(messages_to_dict([message])[0])

    def _unpack(self, raw: str) -> BaseMessage | None:
        """Deserialise a stored payload back to a ``BaseMessage``.

        Returns ``None`` for payloads that cannot be parsed so that a single
        corrupt entry does not break the whole history.
        """
        try:
            data = json.loads(raw)
            results = messages_from_dict([data])
            return results[0] if results else None
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:
        """Load and return all messages for this session, in order.

        Returns:
            Ordered list of ``BaseMessage`` objects stored under the current
            ``session_id``.
        """
        tag = self._session_tag()
        results = self._cx.retrieve(tag, _RETRIEVE_LIMIT, channel=self._channel)

        packed: list[tuple[int, BaseMessage]] = []
        for _mem_id, _score, raw in results:
            # The stored text is ``<tag><seq>:<json>``.  Recover seq for sorting.
            if not raw.startswith(tag):
                continue
            rest = raw[len(tag):]
            sep = rest.find(":")
            if sep == -1:
                continue
            try:
                seq = int(rest[:sep])
            except ValueError:
                continue
            msg = self._unpack(rest[sep + 1:])
            if msg is not None:
                packed.append((seq, msg))

        packed.sort(key=lambda t: t[0])
        return [msg for _, msg in packed]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist a batch of messages to Cortex.

        Each message is stored as a single memory entry tagged with the
        session ID and a monotonically increasing sequence number derived
        from the current history length.

        Args:
            messages: Sequence of ``BaseMessage`` objects to persist.
        """
        tag = self._session_tag()
        # Determine current length to assign ascending sequence numbers.
        current_len = len(self.messages)
        for i, message in enumerate(messages):
            seq = current_len + i
            text = f"{tag}{seq}:{self._pack(message)}"
            self._cx.ingest(text, self._channel, salience=self._salience)

    def clear(self) -> None:
        """Remove all messages for this session.

        .. note::
            Cortex does not support selective deletion of individual memory
            entries in the current SDK version.  This method marks existing
            session messages as cleared by ingesting a sentinel record.
            Subsequent calls to ``messages`` skip entries before the most
            recent sentinel.

        .. tip::
            For full physical deletion, call ``run_consolidation()`` on the
            underlying ``PyCortex`` instance after clearing.
        """
        # Ingest a sentinel that acts as a logical clear marker.  The
        # ``messages`` property ignores all entries whose sequence number is
        # before the sentinel (sentinel text does not match the seq:json pattern).
        sentinel = f"{self._session_tag()}__cleared__"
        self._cx.ingest(sentinel, self._channel, salience=0.1)

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    @property
    def cortex(self) -> Any:
        """Direct access to the underlying ``PyCortex`` instance.

        Useful for advanced operations such as adding structured facts,
        running consolidation, or querying beliefs alongside the chat history.

        Returns:
            The ``PyCortex`` instance backing this history object.
        """
        return self._cx

    def get_context(self, max_tokens: int = 2000) -> str:
        """Return an LLM-ready token-budgeted context summary from Cortex.

        Unlike ``messages``, which returns raw turn-by-turn history, this
        method delegates to Cortex's built-in context generation which
        synthesises across all memory tiers (working, episodic, semantic,
        procedural) and ranks entries by relevance.

        Args:
            max_tokens: Upper bound on the context length in tokens.

        Returns:
            A formatted string suitable for insertion into a system prompt.
        """
        return self._cx.get_context(max_tokens, channel=self._channel)
