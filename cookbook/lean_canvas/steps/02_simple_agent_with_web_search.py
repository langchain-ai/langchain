from dotenv import find_dotenv, load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_gigachat import GigaChat
from langgraph.prebuilt import create_react_agent

load_dotenv(find_dotenv())

llm = GigaChat(model="GigaChat-2-Max", top_p=0)

search_tool = DuckDuckGoSearchRun()

agent = create_react_agent(llm, tools=[search_tool], prompt="Ты полезный ассистент")

inputs = {"messages": [("user", "Что такое lean canvas?")]}
messages = agent.invoke(inputs)["messages"]

print(messages[-1].content)
