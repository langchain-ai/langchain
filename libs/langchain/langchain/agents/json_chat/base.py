from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.tools.render import render_text_description


def create_json_chat_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    stop_sequence: bool = True,
) -> Runnable:
    """Create an agent that uses JSON to format its logic, build for Chat Models.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more.
        stop_sequence: Adds a stop token of "Observation:" to avoid hallucinates. 
            Default is True. You may to set this to False if the LLM you are using
            does not support stop sequences.
    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_json_chat_agent

            prompt = hub.pull("hwchase17/react-chat-json")
            model = ChatOpenAI()
            tools = ...

            agent = create_json_chat_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

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
    
        The prompt must have input keys:
            * `tools`: contains descriptions and arguments for each tool.
            * `tool_names`: contains all tool names.
            * `agent_scratchpad`: must be a MessagesPlaceholder. Contains previous agent actions and tool outputs as messages.
        
        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            system = '''Assistant is a large language model trained by OpenAI.

            Assistant is designed to be able to assist with a wide range of tasks, from answering \
            simple questions to providing in-depth explanations and discussions on a wide range of \
            topics. As a language model, Assistant is able to generate human-like text based on \
            the input it receives, allowing it to engage in natural-sounding conversations and \
            provide responses that are coherent and relevant to the topic at hand.

            Assistant is constantly learning and improving, and its capabilities are constantly \
            evolving. It is able to process and understand large amounts of text, and can use this \
            knowledge to provide accurate and informative responses to a wide range of questions. \
            Additionally, Assistant is able to generate its own text based on the input it \
            receives, allowing it to engage in discussions and provide explanations and \
            descriptions on a wide range of topics.

            Overall, Assistant is a powerful system that can help with a wide range of tasks \
            and provide valuable insights and information on a wide range of topics. Whether \
            you need help with a specific question or just want to have a conversation about \
            a particular topic, Assistant is here to assist.'''
            
            human = '''TOOLS
            ------
            Assistant can ask the user to use tools to look up information that may be helpful in \
            answering the users original question. The tools the human can use are:

            {tools}

            RESPONSE FORMAT INSTRUCTIONS
            ----------------------------

            When responding to me, please output a response in one of two formats:

            **Option 1:**
            Use this if you want the human to use a tool.
            Markdown code snippet formatted in the following schema:

            ```json
            {{
                "action": string, \ The action to take. Must be one of {tool_names}
                "action_input": string \ The input to the action
            }}
            ```

            **Option #2:**
            Use this if you want to respond directly to the human. Markdown code snippet formatted \
            in the following schema:

            ```json
            {{
                "action": "Final Answer",
                "action_input": string \ You should put what you want to return to use here
            }}
            ```

            USER'S INPUT
            --------------------
            Here is the user's input (remember to respond with a markdown code snippet of a json \
            blob with a single action, and NOTHING else):

            {input}'''
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", human),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
    """  # noqa: E501
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        llm_to_use = llm.bind(stop=["\nObservation"])
    else:
        llm_to_use = llm

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["intermediate_steps"], template_tool_response=TEMPLATE_TOOL_RESPONSE
            )
        )
        | prompt
        | llm_to_use
        | JSONAgentOutputParser()
    )
    return agent
