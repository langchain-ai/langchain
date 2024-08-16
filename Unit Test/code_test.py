from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_experimental.tools import PythonREPLTool
from llm_factory import llm
from langchain_core.messages import BaseMessage, HumanMessage


def create_agent(llm: AzureChatOpenAI, tools: list, system_prompt: str):
    """
    Creates an agent with the given LLM, tools, and system prompt.

    Args:
        llm (AzureChatOpenAI): The LLM used by the agent.
        tools (list): The list of tools used by the agent.
        system_prompt (str): The system prompt for the agent.

    Returns:
        AgentExecutor: The created agent executor.

    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

def get_code_agent():
    python_repl_tool = PythonREPLTool()

    system_prompt_coder = "You a coding expert. You can generate python code to analyze data and generate charts using matplotlib. You can also wirte code for mathematical calculations. Once code is generated you use the tools to execute them as well"

    code_agent = create_agent(
        llm,
        [python_repl_tool],
        system_prompt_coder,
    )
    return code_agent


def run_coding_agent(prompt):
    code_agent = get_code_agent()
    code_agent.invoke({"messages": [HumanMessage(content=prompt)]})


if __name__ == '__main__':
    prompt = "what are the squares of 5677234 and 676"
    try:
        result = run_coding_agent(prompt)
        print("Unit test PASSED\n ------")
        print(result)
    except:
        print("Unit test FAILED")
