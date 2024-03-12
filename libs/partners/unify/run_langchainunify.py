import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         SystemMessagePromptTemplate)

from langchain_unify import ChatUnify

chat = ChatUnify(unify_api_key="U3xZq5M5A5HvMJ6Vw6jM-STyPOW3pGmZ0pEIvYIvbig=")
messages = [
    {
        "role": "user",
        "content": "Explain who Newton was and his entire theory of gravitation. Give a long detailed response please and explain all of his achievements",
    }
]
# print(chat.invoke(messages))
# print('ran sync')
# async def async_invoke():
#     print(await chat.ainvoke(messages))
#     print('ran async')

# asyncio.run(async_invoke())
for chunk in chat.stream(messages):
    print(chunk.content, end="")
