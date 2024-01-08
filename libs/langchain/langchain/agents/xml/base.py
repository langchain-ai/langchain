from typing import Any, List, Sequence, Tuple, Union

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.agent import BaseSingleActionAgent
from langchain.agents.format_scratchpad import format_xml
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.agents.xml.prompt import agent_instructions
from langchain.callbacks.base import Callbacks
from langchain.chains.llm import LLMChain
from langchain.tools.render import render_text_description


@deprecated("0.1.0", alternative="create_xml_agent", removal="0.2.0")
class XMLAgent(BaseSingleActionAgent):
    """Agent that uses XML tags.

    Args:
        tools: list of tools the agent can choose from
        llm_chain: The LLMChain to call to predict the next action

    Examples:

        .. code-block:: python

            from langchain.agents import XMLAgent
            from langchain

            tools = ...
            model =


    """

    tools: List[BaseTool]
    """List of tools this agent has access to."""
    llm_chain: LLMChain
    """Chain to use to predict action."""

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        base_prompt = ChatPromptTemplate.from_template(agent_instructions)
        return base_prompt + AIMessagePromptTemplate.from_template(
            "{intermediate_steps}"
        )

    @staticmethod
    def get_default_output_parser() -> XMLAgentOutputParser:
        return XMLAgentOutputParser()

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": log,
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = self.llm_chain(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": log,
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = await self.llm_chain.acall(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]


def create_xml_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: BasePromptTemplate
) -> Runnable:
    """Create an agent that uses XML to format its logic.

    Examples:


        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatAnthropic
            from langchain.agents import AgentExecutor, create_xml_agent

            prompt = hub.pull("hwchase17/xml-agent-convo")
            model = ChatAnthropic()
            tools = ...

            agent = create_xml_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Use with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    # Notice that chat_history is a string
                    # since this prompt is aimed at LLMs, not chat models
                    "chat_history": "Human: My name is Bob\nAI: Hello Bob!",
                }
            )

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input keys of
            `tools` and `agent_scratchpad`.

    Returns:
        A runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    """
    missing_vars = {"tools", "agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
    )
    llm_with_stop = llm.bind(stop=["</tool_input>"])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_xml(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | XMLAgentOutputParser()
    )
    return agent
