"""Пример работы с чатом через gigachain """
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat

chat = GigaChat(streaming=True)


for resp in chat.stream([HumanMessage(content="Напиши сочинение про слона")]):
    print(resp.content, end="")
