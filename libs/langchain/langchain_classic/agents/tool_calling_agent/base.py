from collections.abc import Callable, Sequence

from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain_classic.agents.format_scratchpad.tools import (
    format_to_tool_messages,
)
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser

MessageFormatter = Callable[[Sequence[tuple[AgentAction, str]]], list[BaseMessage]]


def create_tool_calling_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    *,
    message_formatter: MessageFormatter = format_to_tool_messages,
) -> Runnable:
    """Create an agent that uses tools.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
        message_formatter: Formatter function to convert (AgentAction, tool output)
            tuples into FunctionMessages.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:
        ```python
        from langchain_classic.agents import (
            AgentExecutor,
            create_tool_calling_agent,
            tool,
        )
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
        model = ChatAnthropic(model="claude-opus-4-1-20250805")

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
        ```

    Prompt:
        The agent prompt must have an `agent_scratchpad` key that is a
            `MessagesPlaceholder`. Intermediate agent actions and tool output
            messages will be passed in here.

    Troubleshooting:
        - If you encounter `invalid_tool_calls` errors, ensure that your tool
          functions return properly formatted responses. Tool outputs should be
          serializable to JSON. For custom objects, implement proper __str__ or
          to_dict methods.
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables),
    )
    if missing_vars:
        msg = f"Prompt missing required variables: {missing_vars}"
        raise ValueError(msg)

    if not hasattr(llm, "bind_tools"):
        msg = "This function requires a bind_tools() method be implemented on the LLM."
        raise ValueError(
            msg,
        )
    llm_with_tools = llm.bind_tools(tools)

    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: message_formatter(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
