import abc
from typing import Generic, TypeVar
from typing import TypedDict, NotRequired, List, Optional, Literal, Set, cast

from langchain_core.callbacks.events import Event
from langchain_core.runnables import RunnableConfig

CallbackEvent = Literal[
    "on_chat_model_start",
    "on_chat_model_end",
    "on_chat_model_error",
    "on_llm_start",
    "on_llm_end",
    "on_llm_error",
    "on_chain_start",
    "on_chain_end",
    "on_chain_error",
    "on_tool_start",
    "on_tool_end",
    "on_tool_error",
    "on_retriever_start",
    "on_retriever_end",
    "on_retriever_error",
    # Where do these come from??!
    "on_prompt_start",
    "on_prompt_end",
    # Streaming events are missing for the most part
    # "on_chain_stream",
    "on_llm_new_token",  # TODO: This should be updated!
]


class BaseCallback(TypedDict):
    """Base event."""

    # id: str  # id for the callback itself
    run_id: str  # id for the run that generated the callback
    tags: NotRequired[List[str]]
    metadata: NotRequired[List[str]]
    parent_id: NotRequired[Optional[str]]
    type: CallbackEvent


T = TypeVar("T", bound=BaseCallback)


def _convert_event_to_callback(
    event: Event,
    *,
    run_id: str,
    tags: Optional[str],
    metadata: Optional,
    parent_id: Optional[str],
) -> BaseCallback:
    """Convert an event to a callback."""
    return cast(
        BaseCallback,
        {
            "run_id": run_id,
            "tags": tags,
            "metadata": metadata,
            "parent_id": parent_id,
            **event,
        },
    )


T = TypeVar("T")


class GenericCallbackHandler(abc.ABC, Generic[T]):
    @abc.abstractmethod
    @property
    def accepts_events(self) -> Optional[Set[CallbackEvent]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def handle_callback(self, callback: BaseCallback) -> T:
        """Handle an event."""

    @abc.abstractmethod
    async def ahandle_callback(self, callback: BaseCallback) -> T:
        """Handle an event asynchronously."""


class CallbackDispatcher(abc.ABC):
    """Interface to dispatch callbacks to all registered handlers."""

    def __init__(
        self,
        *,
        inheritable_callbacks: Optional[List[GenericCallbackHandler]] = None,
        local_callbacks: Optional[List[GenericCallbackHandler]] = None,
        parent_run_id: Optional[str],
        run_id: Optional[str],
    ) -> None:
        self.inheritable_callbacks = inheritable_callbacks or []
        self.local_callbacks = local_callbacks or []
        self.parent_run_id = parent_run_id
        self.run_id = run_id

    def dispatch_event(self, event: Event) -> None:
        """Handle an event."""
        # Delegate to handle event
        callback_with_data = event.copy()
        callback_with_data["run_id"] = self.run_id
        callback_with_data["parent_id"] = self.parent_run_id

        callback_event = _convert_event_to_callback(
            event, run_id=self.run_id, parent_id=self.parent_run_id
        )
        handlers = self.inheritable_callbacks + self.local_callbacks

        for handler in self.inheritable_callbacks:
            if callback_event["type"] not in handler.accepts_events:
                continue
            handler.handle_callback(callback_with_data)

    async def adispatch_event(self, event: Event) -> None:
        """Delegate to the handler"""
        raise NotImplementedError()

    def get_child(
        self,
        *,
        run_id: Optional[str] = None,  # Allow overriding the run_id?
    ) -> "CallbackDispatcher":
        """Get a child."""
        # unpack config and populate stuff from it
        return CallbackDispatcher(
            parent_run_id=self.run_id,
            run_id=run_id,
            inheritable_callbacks=self.inheritable_callbacks,
            local_callbacks=[],  # Populate from config
        )

    @classmethod
    def config(cls, *, config: Optional[RunnableConfig] = None) -> "CallbackDispatcher":
        """Configure callbacks."""
        return cls(parent_run_id=None, run_id=config.run_id)
