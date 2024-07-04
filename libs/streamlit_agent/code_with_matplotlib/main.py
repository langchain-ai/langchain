from typing import Any, Dict, Optional, List, Union
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_models import GigaChat
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
import streamlit as st

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from langchain_experimental.agents.agent_toolkits.gigapython.base import (
    create_code_chat_agent,
)
from code_tool import (
    GigaPythonREPLTool,
)
import matplotlib
import backend_bytesio

matplotlib.use("module://backend_bytesio")

st.title("GigaChain Code Bot")

with st.sidebar:
    st.title("GigaChat Model")
    model = st.selectbox(
        "GIGACHAT_MODEL",
        (
            "GigaChat-Pro",
            "GigaChat",
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
    credentials = st.text_input("GIGACHAT_CREDENTIALS", type="password")
    scope = st.selectbox(
        "GIGACHAT_SCOPE",
        (
            "GIGACHAT_API_PERS",
            "GIGACHAT_API_CORP",
        ),
    )
    st.title("OR")
    user = st.text_input("GIGACHAT_USER")
    password = st.text_input("GIGACHAT_PASSWORD", type="password")

if not credentials and not (user and password):
    st.info("Заполните данные GigaChat для того, чтобы продолжить")
    st.stop()

msgs = StreamlitChatMessageHistory()


class CallbackHandler(BaseCallbackHandler):
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self.status = st.status("Выполняем код")

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        with st.chat_message("assistant"):
            self.message = st.empty()
            self.text = ""

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self.text += token
        self.message.markdown(self.text)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        msgs.add_ai_message(action.log)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        msgs.add_ai_message(finish.log)

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        image = None
        if len(backend_bytesio.images):
            image = backend_bytesio.images[-1]
        backend_bytesio.images = []
        msgs.add_message(
            HumanMessage(
                content=output, additional_kwargs={"is_tool": True, "image": image}
            )
        )
        with self.status:
            st.write(output)
        if image:
            with st.chat_message("assistant"):
                st.image(image)


llm = GigaChat(
    base_url=base_url,
    model=model,
    scope=scope,
    streaming=True,
    credentials=credentials,
    user=user,
    password=password,
    verify_ssl_certs=False,
)

if len(msgs.messages) == 0:
    msgs.add_ai_message("Привет, я ассистент, способный выполнять код")

for msg in msgs.messages:
    if msg.additional_kwargs.get("is_tool", False):
        with st.status("Выполняем код"):
            st.write(msg.content)
        if msg.additional_kwargs.get("image", None):
            with st.chat_message("assistant"):
                st.image(msg.additional_kwargs.get("image"))
    else:
        st.chat_message(msg.type).write(msg.content)

agent = create_code_chat_agent(llm)

python_tool = GigaPythonREPLTool()

agent_executor = AgentExecutor(
    agent=agent,
    tools=[python_tool],
    return_intermediate_steps=True,
    verbose=True,
    max_iterations=5,
)

if prompt := st.chat_input("Ваш вопрос"):
    st.chat_message("human").write(prompt)
    msgs.add_user_message(prompt)
    agent_messages = msgs.messages
    agent_executor.invoke(
        {"input": prompt, "history": agent_messages}, {"callbacks": [CallbackHandler()]}
    )
