"""Generalized chat agent, works with any chat model."""
from __future__ import annotations

from typing import (
    Callable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
    List,
)

from langchain.automaton.runnables import create_llm_program, create_retriever
from langchain.automaton.typedefs import (
    Agent,
    AgentFinish,
    MessageLike,
    RetrievalRequest,
    RetrievalResponse,
)
from langchain.schema import PromptValue, BaseRetriever
from langchain.schema.language_model import (
    BaseLanguageModel,
    LanguageModelInput,
    LanguageModelOutput,
)
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.schema.output_parser import BaseOutputParser
from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
)
from langchain.tools import BaseTool

T = TypeVar("T")


def prompt_generator(input_messages: Sequence[MessageLike]) -> List[BaseMessage]:
    """Generate a prompt from a log of message like objects."""
    messages = []
    for message in input_messages:
        if isinstance(message, BaseMessage):
            messages.append(message)
        elif isinstance(message, RetrievalResponse):
            prompt = ""

            if message.results:
                for idx, doc in enumerate(message.results):
                    prompt += f"--- Result {idx} ---\n"
                    prompt += "Text:\n"
                    # prompt += f"{doc.page_content}"
                    prompt += f"{doc.metadata['title']}"
                    prompt += f"{doc.metadata['description']}"
                    prompt += f"{doc.metadata['source']}"
                    prompt += f"--- End Result {idx} ---\n"
            else:
                prompt = "Found no results for the query."

            messages.append(
                HumanMessage(
                    content="Context: <result>\n" + prompt + "\n</result>",
                )
            )
        else:
            pass
    return messages


class RagAgent(Agent):
    """A generalized chat agent."""

    def __init__(
        self,
        llm: BaseLanguageModel[LanguageModelInput, LanguageModelOutput],
        prompt_generator: Union[
            Callable[
                [Sequence[MessageLike]], Union[str, PromptValue, List[BaseMessage]]
            ],
            Runnable,
        ],
        *,
        tools: Optional[Sequence[BaseTool]] = None,
        stop: Optional[Sequence[str]] = None,
        parser: Union[
            Runnable[Union[BaseMessage, str], MessageLike],
            Callable[[Union[BaseMessage, str]], MessageLike],
            BaseOutputParser,
            None,
        ] = None,
        retriever: Optional[BaseRetriever] = None,
    ) -> None:
        """Initialize the chat agent."""
        invoke_tools = bool(tools)
        self.llm_program = create_llm_program(
            llm,
            prompt_generator=prompt_generator,
            tools=tools,
            parser=parser,
            stop=stop,
            invoke_tools=invoke_tools,
        )
        self.retriever = create_retriever(retriever)

    def step(
        self,
        messages: Sequence[MessageLike],
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[MessageLike]:
        """Take a single step with the agent."""
        last_message = messages[-1] if messages else None
        if not last_message:
            return []
        match last_message:
            case AIMessage():
                return []
            case AgentFinish():
                return []
            case HumanMessage():
                return [RetrievalRequest(query=last_message.content)]
            case RetrievalRequest():
                return [self.retriever.invoke(last_message, config=config)]
            case _:
                return self.llm_program.invoke(messages, config=config)
