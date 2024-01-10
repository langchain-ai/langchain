import os

from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.chains import LLMMathChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.utilities import GoogleSearchAPIWrapper

from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.functions_utils import (
    VertexAIFunctionsAgentOutputParser,
    format_tools_to_vertex_tool,
)


def test_tools() -> None:
    llm = ChatVertexAI(model_name="gemini-pro")
    math_chain = LLMMathChain.from_llm(llm=llm)
    raw_tools = [
        Tool(
            name="Calculator",
            func=math_chain.run,
            description="useful for when you need to answer questions about math",
        )
    ]
    tools = format_tools_to_vertex_tool(raw_tools)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm_with_tools = llm.bind(tools=tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | VertexAIFunctionsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=raw_tools, verbose=True)

    response = agent_executor.invoke({"input": "What is 6 raised to the 0.43 power?"})
    assert isinstance(response, dict)
    assert response["input"] == "What is 6 raised to the 0.43 power?"
    assert round(float(response["output"]), 3) == 2.161


def test_multiple_tools() -> None:
    llm = ChatVertexAI(model_name="gemini-pro", max_output_tokens=1024)
    math_chain = LLMMathChain.from_llm(llm=llm)
    google_search_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]
    google_cse_id = os.environ["GOOGLE_CSE_ID"]
    search = GoogleSearchAPIWrapper(
        k=10, google_api_key=google_search_api_key, google_cse_id=google_cse_id
    )
    raw_tools = [
        Tool(
            name="Calculator",
            func=math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
        Tool(
            name="Search",
            func=search.run,
            description=(
                "useful for when you need to answer questions about current events. "
                "You should ask targeted questions"
            ),
        ),
    ]
    tools = format_tools_to_vertex_tool(raw_tools)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm_with_tools = llm.bind(tools=tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | VertexAIFunctionsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=raw_tools, verbose=True)

    question = (
        "Who is Leo DiCaprio's girlfriend? What is her "
        "current age raised to the 0.43 power?"
    )
    try:
        response = agent_executor.invoke({"input": question})
    except Exception as err:
        print(err)
        print(err.responses[0])
    assert isinstance(response, dict)
    assert response["input"] == question
    assert "3.850" in response["output"]
