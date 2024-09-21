
import os
os.environ["UNIFY_API_KEY"] = ""

from langchain_community.chat_models.unify import ChatUnify
#from langchain_community.chat_models.unifyai import UnifyChat
chat = ChatUnify(model="gpt-3.5-turbo@openai")
print(chat.invoke("Hello! How are you?"))


