from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_openai import ChatOpenAI

chat = ChatOpenAI(temperature=0, openai_api_key="OPENAI_API_KEY")

messages = [
    SystemMessage(
        content="You are a helpful assistant that answers general knowledge questions."
    ),
    HumanMessage(
        content="Answer the following question. How can I help save the world."
    ),
]

# print(chat.invoke(messages))

print("---STARTING OPENAI CALL---")

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)

from libs.community.langchain_community.chat_models.symblai_nebula import ChatNebula

chat = ChatNebula(temperature=0, nebula_api_key="NEBULA_API_KEY")

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant that answers general knowledge questions."
#     ),
#     HumanMessage(
#         content="Answer the following question. Write one paragraph on global warming."
#     ),
# ]

# print(chat.invoke(messages))

print("\n" + "---NOW STARTING NEBULA CALL---")

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)




