from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.tools.render import render_text_description

TEMPLATE_TOOL_RESPONSE = """TOOL RESPONSE: 
---------------------
{observation}

USER'S INPUT
--------------------

Okay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what!"""

def create_json_chat_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
):
    """Create an agent that uses JSON to format its logic, build for Chat Models.

    Examples:


        .. code-block:: python

            from langchain.agents import (
                create_json_chat_agent,
            )
            from langchain import hub
            from langchain.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor

            prompt = hub.pull("hwchase17/react-chat-json")
            model = ChatOpenAI()
            tools = ...

            agent = create_json_chat_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input keys of
            `tools`, `tool_names`, and `agent_scratchpad`.

    Returns:
        A runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    """
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["intermediate_steps"],
                template_tool_response=TEMPLATE_TOOL_RESPONSE
            ),
            # Give it a default
            chat_history=lambda x: x.get("chat_history", ""),
        )
        | prompt
        | llm_with_stop
        | JSONAgentOutputParser()
    )
    return agent
