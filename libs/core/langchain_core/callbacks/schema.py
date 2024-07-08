import abc
from typing import Generic, TypeVar, Dict, Any, Union
from typing import TypedDict, NotRequired, List, Optional, Literal, Set, cast
from uuid import UUID

from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_core.callbacks.events import Event, ChainStartEvent, ChainEndEvent, ChainErrorEvent
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
    # @abc.abstractmethod
    @property
    def accepts_events(self) -> Optional[Set[CallbackEvent]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def handle_callback(self, callback: BaseCallback) -> T:
        """Handle an event."""

    @abc.abstractmethod
    async def ahandle_callback(self, callback: BaseCallback) -> T:
        """Handle an event asynchronously."""


# TODO(Eugene): This inherits from a bunch of stuff
# for backwards compatibility reasons, but prior to merging
# we need to clean a bunch of those stuff.
class CallbackDispatcher(CallbackManager):
    """Interface to dispatch callbacks to all registered handlers."""

    def __init__(
        self,
        *,
        handlers: List[Union[BaseCallbackHandler, GenericCallbackHandler]],
        inheritable_handlers: Optional[
            List[Union[BaseCallbackHandler, GenericCallbackHandler]]
        ] = None,
        parent_run_id: Optional[str],
        run_id: Optional[str],
        tags: Optional[List[str]] = None,
        inheritable_tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inheritable_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            handlers=handlers,
            inheritable_handlers=inheritable_handlers,
            parent_run_id=parent_run_id,
            tags=tags,
            inheritable_tags=inheritable_tags,
            metadata=metadata,
            inheritable_metadata=inheritable_metadata,
        )
        self.run_id = run_id

    def dispatch_event(self, event: Event) -> None:
        """Handle an event."""
        # Delegate to handle event
        callback_event = _convert_event_to_callback(
            event,
            run_id=self.run_id,
            parent_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
        )
        # handle_event(
        #     self.handlers,
        #     event_name=event["type"],
        #     ignore_condition_name=None,
        #     _event=callback_event,
        # )
        # if isinstance(handler, GenericCallbackHandler):
        #     if callback_event["type"] not in handler.accepts_events:
        #         continue
        #     handler.handle_callback(callback_with_data)
        # else:
        #     handler.handle_callback(callback_with_data)
        #

    async def adispatch_event(self, event: Event) -> None:
        """Delegate to the handler"""
        raise NotImplementedError()

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Union[Dict[str, Any], Any],
        **kwargs: Any,
    ) -> None:
        """Handle a chain start event."""
        event: ChainStartEvent = {
            "type": "on_chain_start",
            "serialized": serialized,
            "inputs": inputs,
            "kwargs": kwargs,
        }
        self.dispatch_event(event)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        event: ChainEndEvent = {
            "type": "on_chain_end",
            "kwargs": kwargs
        }
        self.dispatch_event(event)

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        event: ChainErrorEvent = {
            "type": "on_chain_error",
            "error": error,
            "kwargs": kwargs
        }
        self.dispatch_event(event)

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
            inheritable_handlers=self.inheritable_handlers,
            handlers=[],
        )

    @classmethod
    def config(cls, *, config: Optional[RunnableConfig] = None) -> "CallbackDispatcher":
        """Configure callbacks."""
        return cls(parent_run_id=None, run_id=config.run_id)
