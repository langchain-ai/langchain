import os
import re
from typing import List, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from langchain_google_vertexai.chat_models import ChatVertexAI


class _TestOutputParser(BaseOutputParser):
    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        function_call = message.additional_kwargs.get("function_call", {})
        if function_call:
            function_name = function_call["name"]
            tool_input = function_call.get("arguments", {})

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log_msg = (
                f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            )
            return AgentActionMessageLog(
                tool=function_name,
                tool_input=tool_input,
                log=log_msg,
                message_log=[message],
            )

        return AgentFinish(
            return_values={"output": message.content}, log=str(message.content)
        )

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise ValueError("Can only parse messages")


def test_tools() -> None:
    from langchain.agents import AgentExecutor  # type: ignore[import-not-found]
    from langchain.agents.format_scratchpad import (  # type: ignore[import-not-found]
        format_to_openai_function_messages,
    )
    from langchain.chains import LLMMathChain  # type: ignore[import-not-found]

    llm = ChatVertexAI(model_name="gemini-pro")
    math_chain = LLMMathChain.from_llm(llm=llm)
    tools = [
        Tool(
            name="Calculator",
            func=math_chain.run,
            description="useful for when you need to answer questions about math",
        )
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm_with_tools = llm.bind(functions=tools)

    agent = (
        {  # type: ignore[var-annotated]
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | _TestOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": "What is 6 raised to the 0.43 power?"})
    print(response)
    assert isinstance(response, dict)
    assert response["input"] == "What is 6 raised to the 0.43 power?"

    # convert string " The result is 2.160752567226312" to just numbers/periods
    # use regex to find \d+\.\d+
    just_numbers = re.findall(r"\d+\.\d+", response["output"])[0]

    assert round(float(just_numbers), 3) == 2.161


def test_stream() -> None:
    from langchain.chains import LLMMathChain

    llm = ChatVertexAI(model_name="gemini-pro")
    math_chain = LLMMathChain.from_llm(llm=llm)
    tools = [
        Tool(
            name="Calculator",
            func=math_chain.run,
            description="useful for when you need to answer questions about math",
        )
    ]
    response = list(llm.stream("What is 6 raised to the 0.43 power?", functions=tools))
    assert len(response) == 1
    # for chunk in response:
    assert isinstance(response[0], AIMessageChunk)
    assert "function_call" in response[0].additional_kwargs


def test_multiple_tools() -> None:
    from langchain.agents import AgentExecutor
    from langchain.agents.format_scratchpad import format_to_openai_function_messages
    from langchain.chains import LLMMathChain
    from langchain.utilities import (  # type: ignore[import-not-found]
        GoogleSearchAPIWrapper,
    )

    llm = ChatVertexAI(model_name="gemini-pro", max_output_tokens=1024)
    math_chain = LLMMathChain.from_llm(llm=llm)
    google_search_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]
    google_cse_id = os.environ["GOOGLE_CSE_ID"]
    search = GoogleSearchAPIWrapper(
        k=10, google_api_key=google_search_api_key, google_cse_id=google_cse_id
    )
    tools = [
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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm_with_tools = llm.bind(functions=tools)

    agent = (
        {  # type: ignore[var-annotated]
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | _TestOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    question = (
        "Who is Leo DiCaprio's girlfriend? What is her "
        "current age raised to the 0.43 power?"
    )
    response = agent_executor.invoke({"input": question})
    assert isinstance(response, dict)
    assert response["input"] == question

    # xfail: not getting age in search result most of time
    # assert "3.850" in response["output"]
