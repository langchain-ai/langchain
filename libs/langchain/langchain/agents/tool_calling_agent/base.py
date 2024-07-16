from typing import Callable, List, Optional, Sequence, Tuple

from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForChainRun

from langchain.agents.format_scratchpad.tools import (
    format_to_tool_messages,
)
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

MessageFormatter = Callable[[Sequence[Tuple[AgentAction, str]]], List[BaseMessage]]

def _apply_callbacks(component, callbacks):
    if callbacks:
        return component.with_config(callbacks=callbacks)
    return component

def create_tool_calling_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    *,
    message_formatter: MessageFormatter = format_to_tool_messages,
    callbacks: Optional[CallbackManagerForChainRun] = None,
) -> Runnable:
    """Create an agent that uses tools and supports callbacks.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
        message_formatter: Formatter function to convert (AgentAction, tool output)
            tuples into FunctionMessages.
        callbacks: Optional CallbackManagerForChainRun to support tracing and other
            callback functionalities during the agent's execution.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            model = ChatAnthropic(model="claude-3-opus-20240229")

            @tool
            def magic_function(input: int) -> int:
                \"\"\"Applies a magic function to an input.\"\"\"
                return input + 2

            tools = [magic_function]

            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    if not hasattr(llm, "bind_tools"):
        raise ValueError(
            "This function requires a .bind_tools method be implemented on the LLM.",
        )
    llm_with_tools = llm.bind_tools(tools)
    llm_with_tools = _apply_callbacks(llm_with_tools, callbacks)

    runnable_passthrough = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: message_formatter(x["intermediate_steps"])
    )
    runnable_passthrough = _apply_callbacks(runnable_passthrough, callbacks)

    output_parser = ToolsAgentOutputParser()
    output_parser = _apply_callbacks(output_parser, callbacks)

    agent = runnable_passthrough | prompt | llm_with_tools | output_parser

    return agent