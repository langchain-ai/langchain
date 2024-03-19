from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.render import format_tool_to_openai_function
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from neo4j_semantic_layer.information_tool import InformationTool
from neo4j_semantic_layer.memory_tool import MemoryTool
from neo4j_semantic_layer.recommendation_tool import RecommenderTool

llm = ChatOpenAI(temperature=0, model="gpt-4")
tools = [InformationTool(), RecommenderTool(), MemoryTool()]

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that finds information about movies "
            " and recommends them. If tools require follow up questions, "
            "make sure to ask the user for clarification. Make sure to include any "
            "available options that need to be clarified in the follow up questions",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: (
            _format_chat_history(x["chat_history"]) if x.get("chat_history") else []
        ),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


# Add typing for input
class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools).with_types(
    input_type=AgentInput
)
