from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from langchain.load.serializable import Serializable
from langchain.schema.storage import BaseStore
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from langchain.schema.runnable import (
    GetLocalVar,
    PutLocalVar,
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)


class BaseMemory(Serializable, ABC):
    """Abstract base class for memory in Chains.

    Memory refers to state in Chains. Memory can be used to store information about
        past executions of a Chain and inject that information into the inputs of
        future executions of the Chain. For example, for conversational Chains Memory
        can be used to store conversations and automatically add them to future model
        prompts so that the model has the necessary context to respond coherently to
        the latest input.

    Example:
        .. code-block:: python

            class SimpleMemory(BaseMemory):
                memories: Dict[str, Any] = dict()

                @property
                def memory_variables(self) -> List[str]:
                    return list(self.memories.keys())

                def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                    return self.memories

                def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                    pass

                def clear(self) -> None:
                    pass
    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """The string keys this memory class will add to chain inputs."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this chain run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""


class BaseMemorySessionManager(Runnable[str, BaseMemory]):
    """"""

    def __init__(
        self,
        store: BaseStore[str, BaseMemory],
        *,
        default: Optional[BaseMemory] = None,
        default_factory: Optional[Callable[[], BaseMemory]] = None,
    ) -> None:
        self._store = store
        self._default = default
        self._default_factory = default_factory

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> BaseMemory:
        return self.mset_default((input))[0]

    def batch(
        self,
        inputs: List[str],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs: Optional[Any],
    ) -> List[BaseMemory]:
        return self.mset_default(inputs)

    def mset_default(self, keys: Sequence[str]) -> List[BaseMemory]:
        vals = self._store.mget(keys)
        missing_idxs = [i for i, v in enumerate(vals) if v is None]
        for i in missing_idxs:
            vals[i] = self.default_factory()
        self._store.mset(((keys[i], vals[i]) for i in missing_idxs))
        return vals

    def default_factory(self) -> BaseMemory:
        return self._default or self._default_factory()


def _get_session_id_default(input: Any) -> str:
    if isinstance(input, dict) and "session_id" in input:
        return input["session_id"]
    return str(uuid.uuid4())


def _pop_session_id(input: Any) -> str:
    if isinstance(input, dict) and "session_id" in input:
        input.pop("session_id")
    return input


def _get_session_id(input: Any) -> str:
    return input["session_id"]


def mem_loader(session_manager: "BaseMemorySessionManager", input_key: str) -> Runnable:
    def _load_fn(x: Dict) -> Dict[str, Any]:
        inputs = x[input_key]
        mem = x["memory"]
        return {**mem.load_memory_variables(inputs), **inputs}

    return (
        {"session_id": _get_session_id_default, input_key: _pop_session_id}
        | PutLocalVar(("session_id", input_key))
        | _get_session_id
        | session_manager
        | GetLocalVar(input_key, passthrough_key="memory")
        | _load_fn
    )


def mem_saver(session_manager: "BaseMemorySessionManager", input_key: str) -> Runnable:
    def _runnable_save(x: Dict[str, Any]) -> Dict[str, Any]:
        mem = x["memory"]
        input = x[input_key]
        if isinstance(x["output"], dict):
            output = x["output"]
        else:
            output = {"output": x["output"]}
        mem.save_context(input, output)
        return x["output"]

    return (
        {
            input_key: GetLocalVar(input_key),
            "memory": GetLocalVar("session_id") | session_manager,
            "output": RunnablePassthrough(),
        }
        | RunnableLambda(_runnable_save)
        | {"output": RunnablePassthrough(), "session_id": GetLocalVar("session_id")}
    )


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    See `ChatMessageHistory` for default implementation.

    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]
    """A list of Messages stored in-memory."""

    def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        self.add_message(AIMessage(content=message))

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""
