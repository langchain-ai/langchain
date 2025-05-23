from langchain_gigachat import GigaChat
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = GigaChat(model="GigaChat-2-Max", top_p=0)
print(llm.invoke("Кто тебя создал?").content)
