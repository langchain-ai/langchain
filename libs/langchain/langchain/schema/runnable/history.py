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
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.schema.runnable.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from langchain.callbacks.tracers.schemas import Run
    from langchain.schema.messages import BaseMessage
    from langchain.schema.runnable.config import RunnableConfig

MessagesOrDictWithMessages = Union[Sequence["BaseMessage"], Dict[str, Any]]
GetSessionHistoryCallable = Callable[..., BaseChatMessageHistory]


class RunnableWithMessageHistory(RunnableBindingBase):
    """A runnable that manages chat message history for another runnable.

    Base runnable must have inputs and outputs that can be converted to a list of
        BaseMessages.

    RunnableWithMessageHistory must always be called with a config that contains session_id, e.g.:
        ``{"configurable": {"session_id": "<SESSION_ID>"}}``

    Example (dict input):
        .. code-block:: python

            from typing import Optional

            from langchain.chat_models import ChatAnthropic
            from langchain.memory.chat_message_histories import RedisChatMessageHistory
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain.schema.runnable.history import RunnableWithMessageHistory


            prompt = ChatPromptTemplate.from_messages([
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            chain = prompt | ChatAnthropic(model="claude-2")

            chain_with_history = RunnableWithMessageHistory(
                chain,
                RedisChatMessageHistory,
                input_messages_key="question",
                history_messages_key="history",
            )

            chain_with_history.invoke(
                {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"session_id": "foo"}}
            )
            # -> "Cosine is ..."
            chain_with_history.invoke(
                {"ability": "math", "question": "What's its inverse"},
                config={"configurable": {"session_id": "foo"}}
            )
            # -> "The inverse of cosine is called arccosine ..."

    """  # noqa: E501

    get_session_history: GetSessionHistoryCallable
    input_messages_key: Optional[str] = None
    output_messages_key: Optional[str] = None
    history_messages_key: Optional[str] = None

    def __init__(
        self,
        runnable: Runnable[
            MessagesOrDictWithMessages,
            Union[str, BaseMessage, MessagesOrDictWithMessages],
        ],
        get_session_history: GetSessionHistoryCallable,
        *,
        input_messages_key: Optional[str] = None,
        output_messages_key: Optional[str] = None,
        history_messages_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RunnableWithMessageHistory.

        Args:
            runnable: The base Runnable to be wrapped.

                Must take as input one of:
                - A sequence of BaseMessages
                - A dict with one key for all messages
                - A dict with one key for the current input string/message(s) and
                    a separate key for historical messages. If the input key points
                    to a string, it will be treated as a HumanMessage in history.

                Must return as output one of:
                - A string which can be treated as an AIMessage
                - A BaseMessage or sequence of BaseMessages
                - A dict with a key for a BaseMessage or sequence of BaseMessages

            get_session_history: Function that returns a new BaseChatMessageHistory
                given a session id. Should take a single
                positional argument `session_id` which is a string and a named argument
                `user_id` which can be a string or None. e.g.:

                ```python
                def get_session_history(
                    session_id: str,
                    *,
                    user_id: Optional[str]=None
                ) -> BaseChatMessageHistory:
                  ...
                ```

            input_messages_key: Must be specified if the base runnable accepts a dict
                as input.
            output_messages_key: Must be specified if the base runnable returns a dict
                as output.
            history_messages_key: Must be specified if the base runnable accepts a dict
                as input and expects a separate key for historical messages.
            **kwargs: Arbitrary additional kwargs to pass to parent class
                ``RunnableBindingBase`` init.
        """  # noqa: E501
        history_chain: Runnable = RunnableLambda(
            self._enter_history, self._aenter_history
        ).with_config(run_name="load_history")
        messages_key = history_messages_key or input_messages_key
        if messages_key:
            history_chain = RunnablePassthrough.assign(
                **{messages_key: history_chain}
            ).with_config(run_name="insert_history")
        bound = (
            history_chain | runnable.with_listeners(on_end=self._exit_history)
        ).with_config(run_name="RunnableWithMessageHistory")
        super().__init__(
            get_session_history=get_session_history,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            bound=bound,
            history_messages_key=history_messages_key,
            **kwargs,
        )

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            super().config_specs
            + [
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for a session.",
                    default="",
                ),
            ]
        )

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        super_schema = super().get_input_schema(config)
        if super_schema.__custom_root_type__ is not None:
            from langchain.schema.messages import BaseMessage

            fields: Dict = {}
            if self.input_messages_key and self.history_messages_key:
                fields[self.input_messages_key] = (
                    Union[str, BaseMessage, Sequence[BaseMessage]],
                    ...,
                )
            elif self.input_messages_key:
                fields[self.input_messages_key] = (Sequence[BaseMessage], ...)
            else:
                fields["__root__"] = (Sequence[BaseMessage], ...)
            if self.history_messages_key:
                fields[self.history_messages_key] = (Sequence[BaseMessage], ...)
            return create_model(  # type: ignore[call-overload]
                "RunnableWithChatHistoryInput",
                **fields,
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
        self, output_val: Union[str, BaseMessage, Sequence[BaseMessage], dict]
    ) -> List[BaseMessage]:
        from langchain.schema.messages import BaseMessage

        if isinstance(output_val, dict):
            output_val = output_val[self.output_messages_key or "output"]

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
        hist = config["configurable"]["message_history"]
        # return only historic messages
        if self.history_messages_key:
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
        hist = config["configurable"]["message_history"]

        # Get the input messages
        inputs = load(run.inputs)
        input_val = inputs[self.input_messages_key or "input"]
        input_messages = self._get_input_messages(input_val)

        # Get the output messages
        output_val = load(run.outputs)
        output_messages = self._get_output_messages(output_val)

        for m in input_messages + output_messages:
            hist.add_message(m)

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        # extract session_id
        if "session_id" not in config.get("configurable", {}):
            example_input = {self.input_messages_key: "foo"}
            example_config = {"configurable": {"session_id": "123"}}
            raise ValueError(
                "session_id_id is required."
                " Pass it in as part of the config argument to .invoke() or .stream()"
                f"\neg. chain.invoke({example_input}, {example_config})"
            )
        # attach message_history
        session_id = config["configurable"]["session_id"]
        config["configurable"]["message_history"] = self.get_session_history(session_id)
        return config
