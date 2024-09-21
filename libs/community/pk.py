
import os
os.environ["UNIFY_API_KEY"] = "1a2Yi8+xTGIsQ8bwxgSUhOvztnIhLPgJALzg5Ys98lI="

from langchain_community.chat_models.unify import ChatUnify
#from langchain_community.chat_models.unifyai import UnifyChat
chat = ChatUnify(model="gpt-3.5-turbo@openai")
print(chat.invoke("Hello! How are you?"))

#print()

