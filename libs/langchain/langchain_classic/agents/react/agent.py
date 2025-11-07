from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.tools.render import ToolsRenderer, render_text_description

from langchain_classic.agents import AgentOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser


def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: AgentOutputParser | None = None,
    tools_renderer: ToolsRenderer = render_text_description,
    *,
    stop_sequence: bool | list[str] = True,
) -> Runnable:
    r"""Create an agent that uses ReAct prompting.

    Based on paper "ReAct: Synergizing Reasoning and Acting in Language Models"
    (https://arxiv.org/abs/2210.03629)

    !!! warning
       This implementation is based on the foundational ReAct paper but is older and
       not well-suited for production applications.
       For a more robust and feature-rich implementation, we recommend using the
       `create_react_agent` function from the LangGraph library.
       See the
       [reference doc](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)
       for more information.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more.
        output_parser: AgentOutputParser for parse the LLM output.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM.
        stop_sequence: bool or list of str.
            If `True`, adds a stop token of "Observation:" to avoid hallucinates.
            If `False`, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            You may to set this to False if the LLM you are using
            does not support stop sequences.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Examples:
        ```python
        from langchain_classic import hub
        from langchain_openai import OpenAI
        from langchain_classic.agents import AgentExecutor, create_react_agent

        prompt = hub.pull("hwchase17/react")
        model = OpenAI()
        tools = ...

        agent = create_react_agent(model, tools, prompt)
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
            * `tools`: contains descriptions and arguments for each tool.
            * `tool_names`: contains all tool names.
            * `agent_scratchpad`: contains previous agent actions and tool outputs as a
                string.

        Here's an example:

        ```python
        from langchain_core.prompts import PromptTemplate

        template = '''Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}'''

        prompt = PromptTemplate.from_template(template)
        ```
    """  # noqa: E501
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables),
    )
    if missing_vars:
        msg = f"Prompt missing required variables: {missing_vars}"
        raise ValueError(msg)

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm
    output_parser = output_parser or ReActSingleInputOutputParser()
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | output_parser
    )
