"""Chain that carries on a conversation and calls an LLM."""

from langchain_core._api import deprecated
from langchain_core.prompts import BasePromptTemplate
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_classic.base_memory import BaseMemory
from langchain_classic.chains.conversation.prompt import PROMPT
from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory.buffer import ConversationBufferMemory


@deprecated(
    since="0.2.7",
    alternative="langchain_core.runnables.history.RunnableWithMessageHistory",
    removal="1.0",
)
class ConversationChain(LLMChain):
    """Chain to have a conversation and load context from memory.

    This class is deprecated in favor of `RunnableWithMessageHistory`. Please refer
    to this tutorial for more detail: https://python.langchain.com/docs/tutorials/chatbot/

    `RunnableWithMessageHistory` offers several benefits, including:

    - Stream, batch, and async support;
    - More flexible memory handling, including the ability to manage memory
        outside the chain;
    - Support for multiple threads.

    Below is a minimal implementation, analogous to using `ConversationChain` with
    the default `ConversationBufferMemory`:

        ```python
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_openai import ChatOpenAI


        store = {}  # memory is maintained outside the chain


        def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]


        model = ChatOpenAI(model="gpt-3.5-turbo-0125")

        chain = RunnableWithMessageHistory(model, get_session_history)
        chain.invoke(
            "Hi I'm Bob.",
            config={"configurable": {"session_id": "1"}},
        )  # session_id determines thread
        ```

    Memory objects can also be incorporated into the `get_session_history` callable:

        ```python
        from langchain_classic.memory import ConversationBufferWindowMemory
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_openai import ChatOpenAI


        store = {}  # memory is maintained outside the chain


        def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
                return store[session_id]

            memory = ConversationBufferWindowMemory(
                chat_memory=store[session_id],
                k=3,
                return_messages=True,
            )
            assert len(memory.memory_variables) == 1
            key = memory.memory_variables[0]
            messages = memory.load_memory_variables({})[key]
            store[session_id] = InMemoryChatMessageHistory(messages=messages)
            return store[session_id]


        model = ChatOpenAI(model="gpt-3.5-turbo-0125")

        chain = RunnableWithMessageHistory(model, get_session_history)
        chain.invoke(
            "Hi I'm Bob.",
            config={"configurable": {"session_id": "1"}},
        )  # session_id determines thread
        ```

    Example:
        ```python
        from langchain_classic.chains import ConversationChain
        from langchain_community.llms import OpenAI

        conversation = ConversationChain(llm=OpenAI())
        ```
    """

    memory: BaseMemory = Field(default_factory=ConversationBufferMemory)
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""

    input_key: str = "input"  #: :meta private:
    output_key: str = "response"  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def input_keys(self) -> list[str]:
        """Use this since so some prompt vars come from history."""
        return [self.input_key]

    @model_validator(mode="after")
    def validate_prompt_input_variables(self) -> Self:
        """Validate that prompt input variables are consistent."""
        memory_keys = self.memory.memory_variables
        input_key = self.input_key
        if input_key in memory_keys:
            msg = (
                f"The input key {input_key} was also found in the memory keys "
                f"({memory_keys}) - please provide keys that don't overlap."
            )
            raise ValueError(msg)
        prompt_variables = self.prompt.input_variables
        expected_keys = [*memory_keys, input_key]
        if set(expected_keys) != set(prompt_variables):
            msg = (
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but got {memory_keys} as inputs from "
                f"memory, and {input_key} as the normal input key."
            )
            raise ValueError(msg)
        return self
