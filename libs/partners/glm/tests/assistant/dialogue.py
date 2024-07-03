import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Type, TypeVar, cast

import streamlit as st
import streamlit_antd_components as sac
from streamlit_chatbox import *
from streamlit_extras.bottom_container import bottom
from zhipuai.core._base_models import construct_type

from langchain_glm.agents.zhipuai_all_tools.base import (
    AllToolsAction,
    AllToolsActionToolEnd,
    AllToolsActionToolStart,
    AllToolsFinish,
    AllToolsLLMStatus,
)
from langchain_glm.callbacks.agent_callback_handler import AgentStatus
from tests.assistant.client import ZhipuAIPluginsClient
from tests.assistant.utils import get_img_base64

OutputType = TypeVar(
    "OutputType",
    bound="Union[AllToolsAction,AllToolsActionToolStart,AllToolsActionToolEnd,AllToolsFinish,AllToolsLLMStatus]",
)
chat_box = ChatBox(
    assistant_avatar=get_img_base64(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "chatchat_icon_blue_square_v2.png",
        )
    )
)


def save_session(conv_name: str = None):
    """save session state to chat context"""
    chat_box.context_from_session(
        conv_name, exclude=["selected_page", "prompt", "cur_conv_name"]
    )


def restore_session(conv_name: str = None):
    """restore sesstion state from chat context"""
    chat_box.context_to_session(
        conv_name, exclude=["selected_page", "prompt", "cur_conv_name"]
    )


def rerun():
    """
    save chat context before rerun
    """
    save_session()
    st.rerun()


def get_messages_history(
    history_len: int = 10, content_in_expander: bool = False
) -> List[Dict]:
    """
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    """

    def filter(msg):
        content = [
            x for x in msg["elements"] if x._output_method in ["markdown", "text"]
        ]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    messages = chat_box.filter_history(history_len=history_len, filter=filter)
    if sys_msg := st.session_state.get("system_message"):
        messages = [{"role": "system", "content": sys_msg}] + messages
    return messages


def add_conv(name: str = ""):
    conv_names = chat_box.get_chat_names()
    if not name:
        i = len(conv_names) + 1
        while True:
            name = f"会话{i}"
            if name not in conv_names:
                break
            i += 1
    if name in conv_names:
        sac.alert(
            "创建新会话出错",
            f"该会话名称 “{name}” 已存在",
            color="error",
            closable=True,
        )
    else:
        chat_box.use_chat_name(name)
        st.session_state["cur_conv_name"] = name


def del_conv(name: str = None):
    conv_names = chat_box.get_chat_names()
    name = name or chat_box.cur_chat_name
    if len(conv_names) == 1:
        sac.alert(
            "删除会话出错", f"这是最后一个会话，无法删除", color="error", closable=True
        )
    elif not name or name not in conv_names:
        sac.alert(
            "删除会话出错", f"无效的会话名称：“{name}”", color="error", closable=True
        )
    else:
        chat_box.del_chat_name(name)
        restore_session()
        st.session_state["cur_conv_name"] = chat_box.cur_chat_name


def clear_conv(name: str = None):
    chat_box.reset_history(name=name or None)


def list_tools():
    return {}


def dialogue_page(client: ZhipuAIPluginsClient):
    ctx = chat_box.context
    ctx.setdefault("uid", uuid.uuid4().hex)
    ctx.setdefault("file_chat_id", None)
    ctx.setdefault("temperature", "0.7")
    st.session_state.setdefault("cur_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("last_conv_name", chat_box.cur_chat_name)

    # sac on_change callbacks not working since st>=1.34
    if st.session_state.cur_conv_name != st.session_state.last_conv_name:
        save_session(st.session_state.last_conv_name)
        restore_session(st.session_state.cur_conv_name)
        st.session_state.last_conv_name = st.session_state.cur_conv_name

    @st.experimental_dialog("重命名会话")
    def rename_conversation():
        name = st.text_input("会话名称")
        if st.button("OK"):
            chat_box.change_chat_name(name)
            restore_session()
            st.session_state["cur_conv_name"] = name
            rerun()

    with st.sidebar:
        tab1, _ = st.tabs(["会话设置", "|"])

        with tab1:
            # 会话
            cols = st.columns(3)
            conv_names = chat_box.get_chat_names()

            def on_conv_change():
                print(conversation_name, st.session_state.cur_conv_name)
                save_session(conversation_name)
                restore_session(st.session_state.cur_conv_name)

            conversation_name = sac.buttons(
                conv_names,
                label="当前会话：",
                key="cur_conv_name",
                on_change=on_conv_change,
            )
            chat_box.use_chat_name(conversation_name)
            conversation_id = chat_box.context["uid"]
            if cols[0].button("新建", on_click=add_conv):
                ...
            if cols[1].button("重命名"):
                rename_conversation()
            if cols[2].button("删除", on_click=del_conv):
                ...

    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "请输入对话内容 "

    # chat input
    with bottom():
        cols = st.columns([1, 1, 15])
        prompt = cols[2].chat_input(chat_input_placeholder, key="prompt")
    if prompt:
        history = get_messages_history()
        chat_box.user_say(prompt)

        chat_box.ai_say("正在思考...")

        text = ""
        started = False
        message_id = uuid.uuid4().hex

        metadata = {
            "message_id": message_id,
        }
        function_call = {}
        for item in client.chat(query=prompt, history=history):
            # clear initial message
            if not started:
                chat_box.update_msg("", streaming=False)
                started = True
            if "AllToolsAction" == item["class_name"]:
                cast_type: Type[OutputType] = AllToolsAction
                item = cast(OutputType, construct_type(type_=cast_type, value=item))
                chat_box.insert_msg(f"")

            elif "AllToolsFinish" == item["class_name"]:
                cast_type: Type[OutputType] = AllToolsFinish
                item = cast(OutputType, construct_type(type_=cast_type, value=item))
                finish = True
                chat_box.insert_msg(item.log)
            elif "AllToolsActionToolStart" == item["class_name"]:
                cast_type: Type[OutputType] = AllToolsActionToolStart
                item = cast(OutputType, construct_type(type_=cast_type, value=item))
                formatted_data = {
                    "Function": item.tool,
                    "function_input": item.tool_input,
                }
                formatted_json = json.dumps(
                    formatted_data, indent=2, ensure_ascii=False
                )
                text = """\n```{}\n```\n""".format(formatted_json)
                function_call[item.run_id] = text
                tool_metadata = {
                    "message_id": item.run_id,
                }
                # chat_box.insert_msg(
                #     Markdown(
                #         text,
                #         title=f"正在解读{item.tool}工具输出结果...",
                #         in_expander=True,
                #         expanded=True,
                #         state="running",
                #         metadata=tool_metadata
                #     )
                # )

            elif "AllToolsActionToolEnd" == item["class_name"]:
                cast_type: Type[OutputType] = AllToolsActionToolEnd
                item = cast(OutputType, construct_type(type_=cast_type, value=item))

                text = """\n```\nObservation:\n{}\n```\n""".format(item.tool_output)

                chat_box.insert_msg(
                    Markdown(
                        item.run_id + "\n" + function_call[item.run_id] + "\n" + text,
                        title=f"Function call {item.tool}.",
                        in_expander=True,
                        expanded=True,
                        state="complete",
                    )
                )

            elif "AllToolsLLMStatus" == item["class_name"]:
                cast_type: Type[OutputType] = AllToolsLLMStatus
                item = cast(OutputType, construct_type(type_=cast_type, value=item))
                if item.status == AgentStatus.error:
                    st.error(item.text)
                elif item.status == AgentStatus.chain_start:
                    chat_box.insert_msg(
                        Markdown(
                            text,
                            title=f"chain_start...",
                            in_expander=True,
                            expanded=True,
                            state="complete",
                        )
                    )

                elif item.status == AgentStatus.llm_start:
                    text = item.text or ""
                    chat_box.insert_msg(
                        Markdown(
                            text,
                            title=f"llm_start...",
                            in_expander=True,
                            expanded=True,
                            state="running",
                        )
                    )

                elif item.status == AgentStatus.llm_new_token:
                    text += item.text
                    chat_box.update_msg(text, streaming=True, metadata=metadata)
                elif item.status == AgentStatus.llm_end:
                    chat_box.update_msg(
                        text, streaming=False, expanded=False, state="complete"
                    )
                    chat_box.insert_msg(item.text)
                elif item.status == AgentStatus.chain_end:
                    chat_box.update_msg(
                        item.text,
                        title=f"chain_end...",
                        streaming=False,
                        expanded=False,
                        state="complete",
                    )
                    if not finish:
                        chat_box.insert_msg(item.text)
                else:
                    st.write("item.status :" + item.status + item.text)

    now = datetime.now()
    with tab1:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
            "清空对话",
            use_container_width=True,
        ):
            chat_box.reset_history()
            rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
