from typing import List, Tuple

from ionic_langchain.tool import IonicTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

tools = [IonicTool().tool()]

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-1106", streaming=True)

# You can modify these!
AI_CONTENT = """
I should use the full pdp url that the tool provides me. 
Always include query parameters
"""
SYSTEM_CONTENT = """
You are a shopping assistant. 
You help humans find the best product given their {input}. 
"""
messages = [
    SystemMessage(content=SYSTEM_CONTENT),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=AI_CONTENT),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate.from_messages(messages)
agent = create_openai_tools_agent(llm, tools, prompt)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)
