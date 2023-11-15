from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain.load import load
from langchain.pydantic_v1 import BaseModel, create_model
from langchain.schema.chat_history import BaseChatMessageHistory
from langchain.schema.runnable.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.schema.runnable.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from langchain.callbacks.tracers.schemas import Run
    from langchain.schema.messages import BaseMessage

InputOutputType = Union[str, "BaseMessage", Sequence["BaseMessage"], Dict[str, Any]]


class RunnableWithMessageHistory(RunnableBindingBase):
    """
    A runnable that manages chat message history for another runnable.

    Example:
        .. code-block:: python

            from langchain.schema.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain.chat_models import ChatAnthropic
            from langchain.schema.runnables import RunnableWithMessageHistory

    """

    factory: Callable[[str], BaseChatMessageHistory]
    input_messages_key: Optional[str] = None
    output_messages_key: Optional[str] = None
    message_history_key: Optional[str] = None

    def __init__(
        self,
        runnable: Runnable[InputOutputType, InputOutputType],
        factory: Callable[[str], BaseChatMessageHistory],
        *,
        input_messages_key: Optional[str] = None,
        output_messages_key: Optional[str] = None,
        message_history_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RunnableWithMessageHistory.

        Args:
            runnable: The base Runnable to be wrapped. Should be stateless and take as
                input either a dict with a key for an input message(s) and optionally
                a key for historical messages, or just a message(s). Should return as
                output a(n) message(s) or a dictionary with a key containing a(n)
                message(s).
            factory: Function that returns a new BaseChatMessageHistory given a
                session id. Should take a single parameter `session_id` which is a
                string.
            input_messages_key: The base runnable input key for accepting the latest
                message in the input. Needs to be specified if base runnable input is
                a dictionary.
            output_messages_key: The base runnable output key which should point to a
                sequence of messages in the output. Needs to be specified if base
                runnable output is a dictionary.
            message_history_key: The base runnable history key which should point to a
                sequence of historical messages in the input.
            **kwargs: Arbitrary additional kwargs to pass to parent class
                ``RunnableBindingBase`` init.
        """
        history_chain: Runnable = RunnableLambda(
            self._enter_history, self._aenter_history
        ).with_config(run_name="load_history")
        messages_key = message_history_key or input_messages_key
        if messages_key:
            history_chain = RunnablePassthrough.assign(
                **{messages_key: history_chain}
            ).with_config(run_name="insert_history")
        bound = (
            history_chain | runnable.with_listeners(on_end=self._exit_history)
        ).with_config(run_name="RunnableWithMessageHistory")
        super().__init__(
            factory=factory,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            bound=bound,
            message_history_key=message_history_key,
            **kwargs,
        )

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            super().config_specs
            + [
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for a user.",
                    default=None,
                ),
                ConfigurableFieldSpec(
                    id="thread_id",
                    annotation=str,
                    name="Thread ID",
                    description="Unique identifier for a thread.",
                    default="",
                ),
            ]
        )

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        super_schema = super().get_input_schema(config)
        if super_schema.__custom_root_type__ is not None and self.input_messages_key:
            # The schema is not correct so we'll default to dict with input_messages_key
            return create_model(  # type: ignore[call-overload]
                "RunnableWithChatHistoryInput",
                **{self.input_messages_key: (str, ...)},
            )
        else:
            return super_schema

    def _get_input_messages(
        self, input_val: Union[str, BaseMessage, Sequence[BaseMessage]]
    ) -> List[BaseMessage]:
        from langchain.schema.messages import BaseMessage

        if isinstance(input_val, str):
            from langchain.schema.messages import HumanMessage

            return [HumanMessage(content=input_val)]
        elif isinstance(input_val, BaseMessage):
            return [input_val]
        elif isinstance(input_val, (list, tuple)):
            return list(input_val)
        else:
            raise ValueError(
                f"Expected str, BaseMessage, List[BaseMessage], or Tuple[BaseMessage]. "
                f"Got {input_val}."
            )

    def _get_output_messages(
        self, output_val: Union[str, BaseMessage, Sequence[BaseMessage]]
    ) -> List[BaseMessage]:
        from langchain.schema.messages import BaseMessage

        if isinstance(output_val, str):
            from langchain.schema.messages import AIMessage

            return [AIMessage(content=output_val)]
        elif isinstance(output_val, BaseMessage):
            return [output_val]
        elif isinstance(output_val, (list, tuple)):
            return list(output_val)
        else:
            raise ValueError()

    def _enter_history(self, input: Any, config: RunnableConfig) -> List[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        # return only historic messages
        if self.message_history_key:
            return hist.messages.copy()
        # return all messages
        else:
            input_val = (
                input if not self.input_messages_key else input[self.input_messages_key]
            )
            return hist.messages.copy() + self._get_input_messages(input_val)

    async def _aenter_history(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> List[BaseMessage]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._enter_history, input, config
        )

    def _exit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        # Get the input messages
        inputs = load(run.inputs)
        input_val = inputs[self.input_messages_key or "input"]
        input_messages = self._get_input_messages(input_val)

        # Get the output messages
        outputs = load(run.outputs)
        output_val = outputs[self.output_messages_key or "output"]
        output_messages = self._get_output_messages(output_val)

        for m in input_messages + output_messages:
            hist.add_message(m)

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        # extract thread_id
        config["configurable"] = config.get("configurable", {})
        if "thread_id" not in config["configurable"]:
            from langchain.schema.messages import HumanMessage

            example_input = {self.input_messages_key: HumanMessage(content="foo")}
            example_config = {"configurable": {"thread_id": "123"}}
            raise ValueError(
                "thread_id is required."
                " Pass it in as part of the config argument to .invoke() or .stream()"
                f"\neg. chain.invoke({example_input}, {example_config})"
            )
        # attach message_history
        thread_id = config["configurable"]["thread_id"]
        user_id = config["configurable"].get("user_id")
        session_id = thread_id if user_id is None else f"{user_id}:{thread_id}"
        config["configurable"]["message_history"] = self.factory(
            session_id,
        )
        return config
