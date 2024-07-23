"""Пример работы с чатом через gigachain """

import streamlit as st

# Try demo - https://gigachat-streaming.streamlit.app/

from langchain_community.chat_models import GigaChat
from langchain.schema import ChatMessage

st.title("GigaChain Bot")

with st.sidebar:
    st.title("GIGACHAT API")
    model = st.selectbox(
        "GIGACHAT_MODEL",
        (
            "GigaChat",
            "GigaChat-Pro",
            "GigaChat-Plus",
        ),
    )
    st.title("GIGACHAT API")
    base_url = st.selectbox(
        "GIGACHAT_BASE_URL",
        (
            "https://gigachat.devices.sberbank.ru/api/v1",
            "https://beta.saluteai.sberdevices.ru/v1",
        ),
    )
    st.title("Авторизационные данные")
    scope = st.selectbox(
        "GIGACHAT_SCOPE",
        (
            "GIGACHAT_API_PERS",
            "GIGACHAT_API_CORP",
        ),
    )
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
        ChatMessage(
            role="assistant",
            content="Как я могу помочь вам?",
            additional_kwargs={"render_content": "Как я могу помочь вам?"},
        ),
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.role):
        if message.role == "assistant":
            st.markdown(message.additional_kwargs["render_content"], True)
        else:
            st.markdown(message.content, True)

if prompt := st.chat_input():
    if not access_token and not credentials and not (user and password):
        st.info("Заполните данные GigaChat для того, чтобы продолжить")
        st.stop()

    chat = GigaChat(
        base_url=base_url,
        credentials=credentials,
        model=model,
        access_token=st.session_state.get("token")
        or access_token,  # Переиспользуем токен
        user=user,
        password=password,
        scope=scope,
        verify_ssl_certs=False,
    ).bind_tools(tools=[], tool_choice="auto")

    message = ChatMessage(role="user", content=prompt)
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        st.markdown(message.content)

    message = ChatMessage(
        role="assistant", content="", additional_kwargs={"render_content": ""}
    )
    st.session_state.messages.append(message)

    with st.chat_message(message.role):
        message_placeholder = st.empty()
        spinner = None
        for chunk in chat.stream(st.session_state.messages):
            if chunk.type == "FunctionInProgressMessage":
                if spinner is None:
                    spinner = st.spinner(text="")
                    spinner.__enter__()  # Тут хак для показа спинера в streamlit
                    # не знаю долго ли он проработает, но он здесь просто для красоты
                continue
            else:
                if spinner is not None:
                    spinner.__exit__(None, None, None)
                    spinner = None
                if chunk.additional_kwargs.get("image_uuid"):
                    image_uuid = chunk.additional_kwargs.get("image_uuid")
                    message.additional_kwargs[
                        "render_content"
                    ] += f"""<img src="data:png;base64,{chat.get_file(image_uuid).content}" style="width: 450px; display: block; border-radius: 10px;" >"""
                else:
                    message.additional_kwargs["render_content"] += chunk.content
            message.content += chunk.content
            message.additional_kwargs = {
                **message.additional_kwargs,
                **chunk.additional_kwargs,
            }

            message_placeholder.markdown(
                message.additional_kwargs["render_content"] + "▌", True
            )
        message_placeholder.markdown(message.additional_kwargs["render_content"], True)

    # Каждый раз, когда пользователь нажимает что-то в интерфейсе весь скрипт выполняется заново.
    # Сохраняем токен и закрываем соединения
    st.session_state.token = chat._client.token
    chat._client.close()
