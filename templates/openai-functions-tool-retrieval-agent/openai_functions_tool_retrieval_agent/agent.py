from typing import Dict, List, Tuple

from langchain.agents import (
    AgentExecutor,
    Tool,
)
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.runnable import Runnable, RunnableLambda, RunnableParallel
from langchain.tools.base import BaseTool
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.vectorstores import FAISS

# Create the tools
search = TavilySearchAPIWrapper()
description = """"Useful for when you need to answer questions \
about current events or about recent information."""
tavily_tool = TavilySearchResults(api_wrapper=search, description=description)


def fake_func(inp: str) -> str:
    return "foo"


fake_tools = [
    Tool(
        name=f"foo-{i}",
        func=fake_func,
        description=("a silly function that gets info " f"about the number {i}"),
    )
    for i in range(99)
]
ALL_TOOLS: List[BaseTool] = [tavily_tool] + fake_tools

# turn tools into documents for indexing
docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(ALL_TOOLS)
]

vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

retriever = vector_store.as_retriever()


def get_tools(query: str) -> List[Tool]:
    docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in docs]


assistant_system_message = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""
assistant_system_message = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def llm_with_tools(input: Dict) -> Runnable:
    return RunnableLambda(lambda x: x["input"]) | ChatOpenAI(temperature=0).bind(
        functions=input["functions"]
    )


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    RunnableParallel(
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "agent_scratchpad": lambda x: format_to_openai_functions(
                x["intermediate_steps"]
            ),
            "functions": lambda x: [
                format_tool_to_openai_function(tool) for tool in get_tools(x["input"])
            ],
        }
    )
    | {
        "input": prompt,
        "functions": lambda x: x["functions"],
    }
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# LLM chain consisting of the LLM and a prompt


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=ALL_TOOLS).with_types(
    input_type=AgentInput
)
