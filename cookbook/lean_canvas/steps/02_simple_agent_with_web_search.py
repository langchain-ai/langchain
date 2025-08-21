from dotenv import find_dotenv, load_dotenv
from ddgs import DDGS
from langchain.tools import tool
from langchain_gigachat import GigaChat
from langgraph.prebuilt import create_react_agent

load_dotenv(find_dotenv())

llm = GigaChat(model="GigaChat-2-Max", top_p=0)

@tool("search_tool", description="Ищет в поисковике (RU, неделя, 5 ссылок)")
def search_tool(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        hits = ddgs.text(query, region="ru-ru", time="w", max_results=max_results)
        return "\n".join(f"{hit['title']}: {hit['body']} -- {hit['href']}" for hit in hits[:max_results])

agent = create_react_agent(llm, tools=[search_tool], prompt="Ты полезный ассистент")

inputs = {"messages": [("user", "Что такое lean canvas?")]}
messages = agent.invoke(inputs)["messages"]

print(messages[-1].content)
