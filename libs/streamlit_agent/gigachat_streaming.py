"""Пример работы с чатом через gigachain """
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


with st.sidebar:
    giga_user = st.text_input("Giga User")
    giga_password = st.text_input("Giga Password", type="password")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="system",
            content="Ты - умный ИИ ассистент, \
который всегда готов помочь пользователю.",
        ),
        ChatMessage(role="assistant", content="Как я могу помочь вам?"),
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    if not giga_user or not giga_password:
        st.info("Заполните данные GigaChat для того, чтобы продолжить")
        st.stop()
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = GigaChat(
            user=giga_user,
            password=giga_password,
            streaming=True,
            callbacks=[stream_handler],
        )
        response = llm(st.session_state.messages)
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response.content)
        )
