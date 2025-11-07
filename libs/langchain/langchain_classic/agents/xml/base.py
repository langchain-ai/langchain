from collections.abc import Sequence
from typing import Any

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.tools.render import ToolsRenderer, render_text_description
from typing_extensions import override

from langchain_classic.agents.agent import BaseSingleActionAgent
from langchain_classic.agents.format_scratchpad import format_xml
from langchain_classic.agents.output_parsers import XMLAgentOutputParser
from langchain_classic.agents.xml.prompt import agent_instructions
from langchain_classic.chains.llm import LLMChain


@deprecated("0.1.0", alternative="create_xml_agent", removal="1.0")
class XMLAgent(BaseSingleActionAgent):
    """Agent that uses XML tags.

    Args:
        tools: list of tools the agent can choose from
        llm_chain: The LLMChain to call to predict the next action

    Examples:
        ```python
        from langchain_classic.agents import XMLAgent
        from langchain

        tools = ...
        model =

        ```
    """

    tools: list[BaseTool]
    """List of tools this agent has access to."""
    llm_chain: LLMChain
    """Chain to use to predict action."""

    @property
    @override
    def input_keys(self) -> list[str]:
        return ["input"]

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        """Return the default prompt for the XML agent."""
        base_prompt = ChatPromptTemplate.from_template(agent_instructions)
        return base_prompt + AIMessagePromptTemplate.from_template(
            "{intermediate_steps}",
        )

    @staticmethod
    def get_default_output_parser() -> XMLAgentOutputParser:
        """Return an XMLAgentOutputParser."""
        return XMLAgentOutputParser()

    @override
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
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

    @override
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
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
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: bool | list[str] = True,
) -> Runnable:
    r"""Create an agent that uses XML to format its logic.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input keys
            `tools`: contains descriptions for each tool.
            `agent_scratchpad`: contains previous agent actions and tool outputs.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM.
        stop_sequence: bool or list of str.
            If `True`, adds a stop token of "</tool_input>" to avoid hallucinates.
            If `False`, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            You may to set this to False if the LLM you are using
            does not support stop sequences.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:
        ```python
        from langchain_classic import hub
        from langchain_anthropic import ChatAnthropic
        from langchain_classic.agents import AgentExecutor, create_xml_agent

        prompt = hub.pull("hwchase17/xml-agent-convo")
        model = ChatAnthropic(model="claude-3-haiku-20240307")
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
        ```

    Prompt:

        The prompt must have input keys:
            * `tools`: contains descriptions for each tool.
            * `agent_scratchpad`: contains previous agent actions and tool outputs as
              an XML string.

        Here's an example:

        ```python
        from langchain_core.prompts import PromptTemplate

        template = '''You are a helpful assistant. Help the user answer any questions.

        You have access to the following tools:

        {tools}

        In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
        For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

        <tool>search</tool><tool_input>weather in SF</tool_input>
        <observation>64 degrees</observation>

        When you are done, respond with a final answer between <final_answer></final_answer>. For example:

        <final_answer>The weather in SF is 64 degrees</final_answer>

        Begin!

        Previous Conversation:
        {chat_history}

        Question: {input}
        {agent_scratchpad}'''
        prompt = PromptTemplate.from_template(template)
        ```
    """  # noqa: E501
    missing_vars = {"tools", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables),
    )
    if missing_vars:
        msg = f"Prompt missing required variables: {missing_vars}"
        raise ValueError(msg)

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
    )

    if stop_sequence:
        stop = ["</tool_input>"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm

    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_xml(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | XMLAgentOutputParser()
    )
