"""Пример работы с чатом через gigachain """
# from langchain.schema import HumanMessage, SystemMessage
# from langchain.chat_models.gigachat import GigaChat
#
# chat = GigaChat(streaming=True)
#
#
# message = chat([HumanMessage(content="Напиши сочинение про слона")])
# print(message)

import langchain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import GigaChat
from langchain.schema import ChatMessage
import streamlit as st


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        # llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
        llm = GigaChat(streaming=True, callbacks=[stream_handler])
        response = llm(st.session_state.messages)
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response.content))
