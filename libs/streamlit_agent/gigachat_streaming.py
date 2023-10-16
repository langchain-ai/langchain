"""Пример работы с чатом через gigachain """
import streamlit as st

# Try demo - https://gigachat-streaming.streamlit.app/

from langchain.chat_models import GigaChat
from langchain.schema import ChatMessage

st.title("GigaChain Bot")

with st.sidebar:
    st.title("GIGACHAT API")
    base_url = st.selectbox(
        "GIGACHAT_BASE_URL",
        (
            "https://gigachat.devices.sberbank.ru/api/v1",
            "https://beta.saluteai.sberdevices.ru/v1",
        ),
    )
    st.title("Авторизационные данные")
    credentials = st.text_input("GIGACHAT_CREDENTIALS", type="password")
    st.title("OR")
    access_token = st.text_input("GIGACHAT_ACCESS_TOKEN", type="password")
    st.title("OR")
    user = st.text_input("GIGACHAT_USER")
    password = st.text_input("GIGACHAT_PASSWORD", type="password")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        ChatMessage(
            role="system",
            content="Ты - умный ИИ ассистент, который всегда готов помочь пользователю.",
        ),
        ChatMessage(role="assistant", content="Как я могу помочь вам?"),
    ]


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)


if prompt := st.chat_input():
    if not access_token and not credentials and not (user and password):
        st.info("Заполните данные GigaChat для того, чтобы продолжить")
        st.stop()

    chat = GigaChat(
        base_url=base_url,
        credentials=credentials,
        access_token=st.session_state.get("token")
        or access_token,  # Переиспользуем токен
        user=user,
        password=password,
        verify_ssl_certs=False,
    )

    message = ChatMessage(role="user", content=prompt)
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        st.markdown(message.content)

    message = ChatMessage(role="assistant", content="")
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        message_placeholder = st.empty()
        for chunk in chat.stream(st.session_state.messages):
            message.content += chunk.content
            message_placeholder.markdown(message.content + "▌")
        message_placeholder.markdown(message.content)

    # Каждый раз, когда пользователь нажимает что-то в интерфейсе весь скрипт выполняется заново.
    # Сохраняем токен и закрываем соединения
    st.session_state.token = chat._client.token
    chat._client.close()
