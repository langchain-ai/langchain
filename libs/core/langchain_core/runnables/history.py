from __future__ import annotations

import inspect
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

from pydantic import BaseModel

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Output,
    get_unique_config_specs,
)
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.language_models.base import LanguageModelLike
    from langchain_core.messages.base import BaseMessage
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tracers.schemas import Run


MessagesOrDictWithMessages = Union[Sequence["BaseMessage"], Dict[str, Any]]
GetSessionHistoryCallable = Callable[..., BaseChatMessageHistory]


class RunnableWithMessageHistory(RunnableBindingBase):
    """Runnable that manages chat message history for another Runnable.

    A chat message history is a sequence of messages that represent a conversation.

    RunnableWithMessageHistory wraps another Runnable and manages the chat message
    history for it; it is responsible for reading and updating the chat message
    history.

    The formats supported for the inputs and outputs of the wrapped Runnable
    are described below.

    RunnableWithMessageHistory must always be called with a config that contains
    the appropriate parameters for the chat message history factory.

    By default, the Runnable is expected to take a single configuration parameter
    called `session_id` which is a string. This parameter is used to create a new
    or look up an existing chat message history that matches the given session_id.

    In this case, the invocation would look like this:

    `with_history.invoke(..., config={"configurable": {"session_id": "bar"}})`
    ; e.g., ``{"configurable": {"session_id": "<SESSION_ID>"}}``.

    The configuration can be customized by passing in a list of
    ``ConfigurableFieldSpec`` objects to the ``history_factory_config`` parameter (see
    example below).

    In the examples, we will use a chat message history with an in-memory
    implementation to make it easy to experiment and see the results.

    For production use cases, you will want to use a persistent implementation
    of chat message history, such as ``RedisChatMessageHistory``.

    Parameters:
        get_session_history: Function that returns a new BaseChatMessageHistory.
            This function should either take a single positional argument
            `session_id` of type string and return a corresponding
            chat message history instance.
        input_messages_key: Must be specified if the base runnable accepts a dict
            as input. The key in the input dict that contains the messages.
        output_messages_key: Must be specified if the base Runnable returns a dict
            as output. The key in the output dict that contains the messages.
        history_messages_key: Must be specified if the base runnable accepts a dict
            as input and expects a separate key for historical messages.
        history_factory_config: Configure fields that should be passed to the
            chat history factory. See ``ConfigurableFieldSpec`` for more details.

    Example: Chat message history with an in-memory implementation for testing.

    .. code-block:: python

        from operator import itemgetter
        from typing import List

        from langchain_openai.chat_models import ChatOpenAI

        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.documents import Document
        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from pydantic import BaseModel, Field
        from langchain_core.runnables import (
            RunnableLambda,
            ConfigurableFieldSpec,
            RunnablePassthrough,
        )
        from langchain_core.runnables.history import RunnableWithMessageHistory


        class InMemoryHistory(BaseChatMessageHistory, BaseModel):
            \"\"\"In memory implementation of chat message history.\"\"\"

            messages: List[BaseMessage] = Field(default_factory=list)

            def add_messages(self, messages: List[BaseMessage]) -> None:
                \"\"\"Add a list of messages to the store\"\"\"
                self.messages.extend(messages)

            def clear(self) -> None:
                self.messages = []

        # Here we use a global variable to store the chat message history.
        # This will make it easier to inspect it to see the underlying results.
        store = {}

        def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryHistory()
            return store[session_id]


        history = get_by_session_id("1")
        history.add_message(AIMessage(content="hello"))
        print(store)  # noqa: T201


    Example where the wrapped Runnable takes a dictionary input:

        .. code-block:: python

            from typing import Optional

            from langchain_community.chat_models import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.runnables.history import RunnableWithMessageHistory


            prompt = ChatPromptTemplate.from_messages([
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            chain = prompt | ChatAnthropic(model="claude-2")

            chain_with_history = RunnableWithMessageHistory(
                chain,
                # Uses the get_by_session_id function defined in the example
                # above.
                get_by_session_id,
                input_messages_key="question",
                history_messages_key="history",
            )

            print(chain_with_history.invoke(  # noqa: T201
                {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"session_id": "foo"}}
            ))

            # Uses the store defined in the example above.
            print(store)  # noqa: T201

            print(chain_with_history.invoke(  # noqa: T201
                {"ability": "math", "question": "What's its inverse"},
                config={"configurable": {"session_id": "foo"}}
            ))

            print(store)  # noqa: T201


    Example where the session factory takes two keys, user_id and conversation id):

        .. code-block:: python

            store = {}

            def get_session_history(
                user_id: str, conversation_id: str
            ) -> BaseChatMessageHistory:
                if (user_id, conversation_id) not in store:
                    store[(user_id, conversation_id)] = InMemoryHistory()
                return store[(user_id, conversation_id)]

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You're an assistant who's good at {ability}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            chain = prompt | ChatAnthropic(model="claude-2")

            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history=get_session_history,
                input_messages_key="question",
                history_messages_key="history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="user_id",
                        annotation=str,
                        name="User ID",
                        description="Unique identifier for the user.",
                        default="",
                        is_shared=True,
                    ),
                    ConfigurableFieldSpec(
                        id="conversation_id",
                        annotation=str,
                        name="Conversation ID",
                        description="Unique identifier for the conversation.",
                        default="",
                        is_shared=True,
                    ),
                ],
            )

            with_message_history.invoke(
                {"ability": "math", "question": "What does cosine mean?"},
                config={"configurable": {"user_id": "123", "conversation_id": "1"}}
            )

    """

    get_session_history: GetSessionHistoryCallable
    input_messages_key: Optional[str] = None
    output_messages_key: Optional[str] = None
    history_messages_key: Optional[str] = None
    history_factory_config: Sequence[ConfigurableFieldSpec]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def __init__(
        self,
        runnable: Union[
            Runnable[
                Union[MessagesOrDictWithMessages],
                Union[str, BaseMessage, MessagesOrDictWithMessages],
            ],
            LanguageModelLike,
        ],
        get_session_history: GetSessionHistoryCallable,
        *,
        input_messages_key: Optional[str] = None,
        output_messages_key: Optional[str] = None,
        history_messages_key: Optional[str] = None,
        history_factory_config: Optional[Sequence[ConfigurableFieldSpec]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RunnableWithMessageHistory.

        Args:
            runnable: The base Runnable to be wrapped. Must take as input one of:
                1. A sequence of BaseMessages
                2. A dict with one key for all messages
                3. A dict with one key for the current input string/message(s) and
                    a separate key for historical messages. If the input key points
                    to a string, it will be treated as a HumanMessage in history.

                Must return as output one of:
                1. A string which can be treated as an AIMessage
                2. A BaseMessage or sequence of BaseMessages
                3. A dict with a key for a BaseMessage or sequence of BaseMessages

            get_session_history: Function that returns a new BaseChatMessageHistory.
                This function should either take a single positional argument
                `session_id` of type string and return a corresponding
                chat message history instance.
                .. code-block:: python

                    def get_session_history(
                        session_id: str,
                        *,
                        user_id: Optional[str]=None
                    ) -> BaseChatMessageHistory:
                      ...

                Or it should take keyword arguments that match the keys of
                `session_history_config_specs` and return a corresponding
                chat message history instance.

                .. code-block:: python

                    def get_session_history(
                        *,
                        user_id: str,
                        thread_id: str,
                    ) -> BaseChatMessageHistory:
                        ...

            input_messages_key: Must be specified if the base runnable accepts a dict
                as input. Default is None.
            output_messages_key: Must be specified if the base runnable returns a dict
                as output. Default is None.
            history_messages_key: Must be specified if the base runnable accepts a dict
                as input and expects a separate key for historical messages.
            history_factory_config: Configure fields that should be passed to the
                chat history factory. See ``ConfigurableFieldSpec`` for more details.
                Specifying these allows you to pass multiple config keys
                into the get_session_history factory.
            **kwargs: Arbitrary additional kwargs to pass to parent class
                ``RunnableBindingBase`` init.
        """
        history_chain: Runnable = RunnableLambda(
            self._enter_history, self._aenter_history
        ).with_config(run_name="load_history")
        messages_key = history_messages_key or input_messages_key
        if messages_key:
            history_chain = RunnablePassthrough.assign(
                **{messages_key: history_chain}
            ).with_config(run_name="insert_history")

        runnable_sync: Runnable = runnable.with_listeners(on_end=self._exit_history)
        runnable_async: Runnable = runnable.with_alisteners(on_end=self._aexit_history)

        def _call_runnable_sync(_input: Any) -> Runnable:
            return runnable_sync

        async def _call_runnable_async(_input: Any) -> Runnable:
            return runnable_async

        bound: Runnable = (
            history_chain
            | RunnableLambda(
                _call_runnable_sync,
                _call_runnable_async,
            ).with_config(run_name="check_sync_or_async")
        ).with_config(run_name="RunnableWithMessageHistory")

        if history_factory_config:
            _config_specs = history_factory_config
        else:
            # If not provided, then we'll use the default session_id field
            _config_specs = [
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for a session.",
                    default="",
                    is_shared=True,
                ),
            ]

        super().__init__(
            get_session_history=get_session_history,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            bound=bound,
            history_messages_key=history_messages_key,
            history_factory_config=_config_specs,
            **kwargs,
        )
        self._history_chain = history_chain

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        """Get the configuration specs for the RunnableWithMessageHistory."""
        return get_unique_config_specs(
            super().config_specs + list(self.history_factory_config)
        )

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        from langchain_core.messages import BaseMessage

        fields: Dict = {}
        if self.input_messages_key and self.history_messages_key:
            fields[self.input_messages_key] = (
                Union[str, BaseMessage, Sequence[BaseMessage]],
                ...,
            )
        elif self.input_messages_key:
            fields[self.input_messages_key] = (Sequence[BaseMessage], ...)
        else:
            return create_model_v2(
                "RunnableWithChatHistoryInput",
                module_name=self.__class__.__module__,
                root=(Sequence[BaseMessage], ...),
            )
        return create_model_v2(  # type: ignore[call-overload]
            "RunnableWithChatHistoryInput",
            field_definitions=fields,
            module_name=self.__class__.__module__,
        )

    @property
    def OutputType(self) -> Type[Output]:
        output_type = self._history_chain.OutputType
        return output_type

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        """Get a pydantic model that can be used to validate output to the Runnable.

        Runnables that leverage the configurable_fields and configurable_alternatives
        methods will have a dynamic output schema that depends on which
        configuration the Runnable is invoked with.

        This method allows to get an output schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A pydantic model that can be used to validate output.
        """
        root_type = self.OutputType

        if inspect.isclass(root_type) and issubclass(root_type, BaseModel):
            return root_type

        return create_model_v2(
            "RunnableWithChatHistoryOutput",
            root=root_type,
            module_name=self.__class__.__module__,
        )

    def _is_not_async(self, *args: Sequence[Any], **kwargs: Dict[str, Any]) -> bool:
        return False

    async def _is_async(self, *args: Sequence[Any], **kwargs: Dict[str, Any]) -> bool:
        return True

    def _get_input_messages(
        self, input_val: Union[str, BaseMessage, Sequence[BaseMessage], dict]
    ) -> List[BaseMessage]:
        from langchain_core.messages import BaseMessage

        # If dictionary, try to pluck the single key representing messages
        if isinstance(input_val, dict):
            if self.input_messages_key:
                key = self.input_messages_key
            elif len(input_val) == 1:
                key = list(input_val.keys())[0]
            else:
                key = "input"
            input_val = input_val[key]

        # If value is a string, convert to a human message
        if isinstance(input_val, str):
            from langchain_core.messages import HumanMessage

            return [HumanMessage(content=input_val)]
        # If value is a single message, convert to a list
        elif isinstance(input_val, BaseMessage):
            return [input_val]
        # If value is a list or tuple...
        elif isinstance(input_val, (list, tuple)):
            # Handle empty case
            if len(input_val) == 0:
                return list(input_val)
            # If is a list of list, then return the first value
            # This occurs for chat models - since we batch inputs
            if isinstance(input_val[0], list):
                if len(input_val) != 1:
                    raise ValueError(
                        f"Expected a single list of messages. Got {input_val}."
                    )
                return input_val[0]
            return list(input_val)
        else:
            raise ValueError(
                f"Expected str, BaseMessage, List[BaseMessage], or Tuple[BaseMessage]. "
                f"Got {input_val}."
            )

    def _get_output_messages(
        self, output_val: Union[str, BaseMessage, Sequence[BaseMessage], dict]
    ) -> List[BaseMessage]:
        from langchain_core.messages import BaseMessage

        # If dictionary, try to pluck the single key representing messages
        if isinstance(output_val, dict):
            if self.output_messages_key:
                key = self.output_messages_key
            elif len(output_val) == 1:
                key = list(output_val.keys())[0]
            else:
                key = "output"
            # If you are wrapping a chat model directly
            # The output is actually this weird generations object
            if key not in output_val and "generations" in output_val:
                output_val = output_val["generations"][0][0]["message"]
            else:
                output_val = output_val[key]

        if isinstance(output_val, str):
            from langchain_core.messages import AIMessage

            return [AIMessage(content=output_val)]
        # If value is a single message, convert to a list
        elif isinstance(output_val, BaseMessage):
            return [output_val]
        elif isinstance(output_val, (list, tuple)):
            return list(output_val)
        else:
            raise ValueError(
                f"Expected str, BaseMessage, List[BaseMessage], or Tuple[BaseMessage]. "
                f"Got {output_val}."
            )

    def _enter_history(self, input: Any, config: RunnableConfig) -> List[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = hist.messages.copy()

        if not self.history_messages_key:
            # return all messages
            input_val = (
                input if not self.input_messages_key else input[self.input_messages_key]
            )
            messages += self._get_input_messages(input_val)
        return messages

    async def _aenter_history(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> List[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = (await hist.aget_messages()).copy()

        if not self.history_messages_key:
            # return all messages
            input_val = (
                input if not self.input_messages_key else input[self.input_messages_key]
            )
            messages += self._get_input_messages(input_val)
        return messages

    def _exit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        # Get the input messages
        inputs = load(run.inputs)
        input_messages = self._get_input_messages(inputs)
        # If historic messages were prepended to the input messages, remove them to
        # avoid adding duplicate messages to history.
        if not self.history_messages_key:
            historic_messages = config["configurable"]["message_history"].messages
            input_messages = input_messages[len(historic_messages) :]

        # Get the output messages
        output_val = load(run.outputs)
        output_messages = self._get_output_messages(output_val)
        hist.add_messages(input_messages + output_messages)

    async def _aexit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        # Get the input messages
        inputs = load(run.inputs)
        input_messages = self._get_input_messages(inputs)
        # If historic messages were prepended to the input messages, remove them to
        # avoid adding duplicate messages to history.
        if not self.history_messages_key:
            historic_messages = await hist.aget_messages()
            input_messages = input_messages[len(historic_messages) :]

        # Get the output messages
        output_val = load(run.outputs)
        output_messages = self._get_output_messages(output_val)
        await hist.aadd_messages(input_messages + output_messages)

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        expected_keys = [field_spec.id for field_spec in self.history_factory_config]

        configurable = config.get("configurable", {})

        missing_keys = set(expected_keys) - set(configurable.keys())
        parameter_names = _get_parameter_names(self.get_session_history)

        if missing_keys and parameter_names:
            example_input = {self.input_messages_key: "foo"}
            example_configurable = {
                missing_key: "[your-value-here]" for missing_key in missing_keys
            }
            example_config = {"configurable": example_configurable}
            raise ValueError(
                f"Missing keys {sorted(missing_keys)} in config['configurable'] "
                f"Expected keys are {sorted(expected_keys)}."
                f"When using via .invoke() or .stream(), pass in a config; "
                f"e.g., chain.invoke({example_input}, {example_config})"
            )

        if len(expected_keys) == 1:
            if parameter_names:
                # If arity = 1, then invoke function by positional arguments
                message_history = self.get_session_history(
                    configurable[expected_keys[0]]
                )
            else:
                if not config:
                    config["configurable"] = {}
                message_history = self.get_session_history()
        else:
            # otherwise verify that names of keys patch and invoke by named arguments
            if set(expected_keys) != set(parameter_names):
                raise ValueError(
                    f"Expected keys {sorted(expected_keys)} do not match parameter "
                    f"names {sorted(parameter_names)} of get_session_history."
                )

            message_history = self.get_session_history(
                **{key: configurable[key] for key in expected_keys}
            )
        config["configurable"]["message_history"] = message_history
        return config


def _get_parameter_names(callable_: GetSessionHistoryCallable) -> List[str]:
    """Get the parameter names of the callable."""
    sig = inspect.signature(callable_)
    return list(sig.parameters.keys())
