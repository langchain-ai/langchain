"""Chain that does self-ask with search."""
from typing import Any, Sequence, Union

from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.utilities.searchapi import SearchApiAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.self_ask_with_search.output_parser import SelfAskOutputParser
from langchain.agents.self_ask_with_search.prompt import PROMPT
from langchain.agents.tools import Tool
from langchain.agents.utils import validate_tools_single_input


class SelfAskWithSearchAgent(Agent):
    """Agent for the self-ask-with-search paper."""

    output_parser: AgentOutputParser = Field(default_factory=SelfAskOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return SelfAskOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        return AgentType.SELF_ASK_WITH_SEARCH

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Prompt does not depend on tools."""
        return PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        super()._validate_tools(tools)
        if len(tools) != 1:
            raise ValueError(f"Exactly one tool must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Intermediate Answer"}:
            raise ValueError(
                f"Tool name should be Intermediate Answer, got {tool_names}"
            )

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Intermediate answer: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return ""


class SelfAskWithSearchChain(AgentExecutor):
    """[Deprecated] Chain that does self-ask with search."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        search_chain: Union[
            GoogleSerperAPIWrapper, SearchApiAPIWrapper, SerpAPIWrapper
        ],
        **kwargs: Any,
    ):
        """Initialize only with an LLM and a search chain."""
        search_tool = Tool(
            name="Intermediate Answer",
            func=search_chain.run,
            coroutine=search_chain.arun,
            description="Search",
        )
        agent = SelfAskWithSearchAgent.from_llm_and_tools(llm, [search_tool])
        super().__init__(agent=agent, tools=[search_tool], **kwargs)


def create_self_ask_with_search_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: BasePromptTemplate
) -> Runnable:
    """Create an agent that uses self-ask with search prompting.

    Examples:


        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatAnthropic
            from langchain.agents import (
                AgentExecutor, create_self_ask_with_search_agent
            )

            prompt = hub.pull("hwchase17/self-ask-with-search")
            model = ChatAnthropic()
            tools = [...]  # Should just be one tool with name `Intermediate Answer`

            agent = create_self_ask_with_search_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

    Args:
        llm: LLM to use as the agent.
        tools: List of tools. Should just be of length 1, with that tool having
            name `Intermediate Answer`
        prompt: The prompt to use, must have input keys of `agent_scratchpad`.

    Returns:
        A runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    """
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    if len(tools) != 1:
        raise ValueError("This agent expects exactly one tool")
    tool = list(tools)[0]
    if tool.name != "Intermediate Answer":
        raise ValueError(
            "This agent expects the tool to be named `Intermediate Answer`"
        )

    llm_with_stop = llm.bind(stop=["\nIntermediate answer:"])
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(
                x["intermediate_steps"],
                observation_prefix="\nIntermediate answer: ",
                llm_prefix="",
            ),
            # Give it a default
            chat_history=lambda x: x.get("chat_history", ""),
        )
        | prompt
        | llm_with_stop
        | SelfAskOutputParser()
    )
    return agent
