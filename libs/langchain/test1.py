from dotenv import load_dotenv
from langchain.chat_models import ChatLiteLLM, ChatAnthropic, ChatGooglePalm

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import LLMonitorCallbackHandler


load_dotenv()

handler = LLMonitorCallbackHandler()

# chat = ChatLiteLLM(model="gpt-3.5-turbo", callbacks=[handler])
# chat = ChatLiteLLM(model="gpt-4", callbacks=[handler], temperature=1)
# chat = ChatAnthropic(model='claude-instant-1', callbacks=[handler])
chat = ChatGooglePalm(callbacks=[handler])



messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]

print(chat(messages))