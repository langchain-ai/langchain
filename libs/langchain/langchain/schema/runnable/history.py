from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Type

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
    input_key: str
    output_key: Optional[str] = None
    history_key: Optional[str] = None

    def __init__(
        self,
        runnable: Runnable[Dict[str, Any], Sequence[BaseMessage]],
        factory: Callable[[str], BaseChatMessageHistory],
        input_key: str,
        *,
        output_key: Optional[str] = None,
        history_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RunnableWithMessageHistory.

        Args:
            runnable: The base Runnable to be wrapped. Should be stateless and take as
                input a dict with a key for an input message(s) and optionally
                a key for historical messages. Should return as output a(n) message(s)
                or a dictionary with a key containing a(n) message(s).
            factory: Function that returns a new BaseChatMessageHistory given a
                session id. Should take a single parameter `session_id` which is a
                string.
            input_key: The base runnable input key for accepting the latest message in
                the input.
            output_key: The base runnable output key which should point to a sequence
                of messages in the output.
            history_key: The base runnable history key which should point to a sequence
                of historical messages in the input.
            **kwargs: Arbitrary additional kwargs to pass to parent class
                ``RunnableBindingBase`` init.
        """
        messages_key = history_key or input_key
        bound = (
            RunnablePassthrough.assign(
                **{
                    messages_key: RunnableLambda(
                        self._enter_history, self._aenter_history
                    ).with_config(run_name="load_history")
                }
            ).with_config(run_name="insert_history")
            | runnable.with_listeners(on_end=self._exit_history)
        ).with_config(run_name="RunnableWithMessageHistory")
        super().__init__(
            factory=factory,
            input_key=input_key,
            output_key=output_key,
            bound=bound,
            history_key=history_key,
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
        if super_schema.__custom_root_type__ is not None:
            # The schema is not correct so we'll default to dict with input_key
            return create_model(  # type: ignore[call-overload]
                "RunnableWithChatHistoryInput",
                **{self.input_key: (str, ...)},
            )
        else:
            return super_schema

    def _get_input_messages(self, input: Dict[str, Any]) -> List[BaseMessage]:
        from langchain.schema.messages import BaseMessage

        input_val = input[self.input_key]
        if isinstance(input_val, str):
            from langchain.schema.messages import HumanMessage

            return [HumanMessage(content=input_val)]
        elif isinstance(input_val, BaseMessage):
            return [input_val]
        elif isinstance(input_val, list):
            return input_val
        else:
            raise ValueError()

    def _enter_history(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> List[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        # return only historic messages
        if self.history_key:
            return hist.messages.copy()
        # return all messages
        else:
            return hist.messages.copy() + self._get_input_messages(input)

    async def _aenter_history(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> List[BaseMessage]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._enter_history, input, config
        )

    def _exit_history(self, run: Run, config: RunnableConfig) -> None:
        from langchain.schema.messages import BaseMessage

        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        # Add the input message
        input_messages = self._get_input_messages(run.inputs)
        for m in input_messages:
            hist.add_message(m)

        # Add the output messages
        outputs = load(run.outputs)
        if self.output_key is not None:
            output_messages: List[BaseMessage] = outputs[self.output_key]
        elif "output" in outputs:
            output_messages = outputs["output"]
        else:
            raise ValueError(
                f"Output is a dict but not output_key was specified. Received output: "
                f"{outputs}"
            )
        if isinstance(output_messages, BaseMessage):
            output_messages = [output_messages]

        for m in output_messages:
            hist.add_message(m)

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        # extract thread_id
        config["configurable"] = config.get("configurable", {})
        if "thread_id" not in config["configurable"]:
            from langchain.schema.messages import HumanMessage

            example_input = {self.input_key: HumanMessage(content="foo")}
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
