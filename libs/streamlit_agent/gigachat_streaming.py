"""Пример работы с чатом через gigachain """
import streamlit as st

# Try demo - https://gigachat-streaming.streamlit.app/

from langchain.chat_models import GigaChat
from langchain.schema import ChatMessage

st.title("GigaChain Bot")

with st.sidebar:
    st.title("GIGACHAT API")
    api_base_url = st.selectbox(
        "GIGA_API_BASE_URL",
        (
            "https://gigachat.devices.sberbank.ru/api/v1",
            "https://beta.saluteai.sberdevices.ru/v1",
        ),
    )
    st.title("Авторизация")
    client_id = st.text_input("GIGA_CLIENT_ID")
    client_secret = st.text_input("GIGA_CLIENT_SECRET", type="password")
    st.title("OR")
    oauth_token = st.text_input("GIGA_OAUTH_TOKEN", type="password")
    st.title("OR")
    user = st.text_input("GIGA_USER")
    password = st.text_input("GIGA_PASSWORD", type="password")
    st.title("OR")
    token = st.text_input("GIGA_TOKEN", type="password")


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
    if (
        not (user and password)
        and not (client_id and client_secret)
        and not oauth_token
        and not token
    ):
        st.info("Заполните данные GigaChat для того, чтобы продолжить")
        st.stop()

    chat = GigaChat(
        api_base_url=api_base_url,
        token=st.session_state.get("token") or token,  # Переиспользуем токен
        user=user,
        password=password,
        client_id=client_id,
        client_secret=client_secret,
        oauth_token=oauth_token,
        verify_ssl=False,
        oauth_verify_ssl=False,
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
