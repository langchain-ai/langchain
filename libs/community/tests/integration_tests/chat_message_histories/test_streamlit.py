"""Unit tests for StreamlitChatMessageHistory functionality."""

import pytest

test_script = """
    import json
    import streamlit as st
    from langchain.memory import ConversationBufferMemory
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory
    from langchain_core.messages import message_to_dict, BaseMessage

    message_history = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(chat_memory=message_history, return_messages=True)

    # Add some messages
    if st.checkbox("add initial messages", value=True):
        memory.chat_memory.add_ai_message("This is me, the AI")
        memory.chat_memory.add_user_message("This is me, the human")
    else:
        st.markdown("Skipped add")

    # Clear messages if checked
    if st.checkbox("clear messages"):
        st.markdown("Cleared!")
        memory.chat_memory.clear()

    # Use message setter
    if st.checkbox("Override messages"):
        memory.chat_memory.messages = [
            BaseMessage(content="A basic message", type="basic")
        ]
        st.session_state["langchain_messages"].append(
            BaseMessage(content="extra cool message", type="basic")
        )

    # Write the output to st.code as a json blob for inspection
    messages = memory.chat_memory.messages
    messages_json = json.dumps([message_to_dict(msg) for msg in messages])
    st.text(messages_json)
"""


@pytest.mark.requires("streamlit")
def test_memory_with_message_store() -> None:
    try:
        from streamlit.testing.v1 import AppTest
    except ModuleNotFoundError:
        pytest.skip("Incorrect version of Streamlit installed")

    at = AppTest.from_string(test_script).run(timeout=10)

    # Initial run should write two messages
    messages_json = at.get("text")[-1].value
    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # Uncheck the initial write, they should persist in session_state
    at.get("checkbox")[0].uncheck().run()
    assert at.get("markdown")[0].value == "Skipped add"
    messages_json = at.get("text")[-1].value
    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # Clear the message history
    at.get("checkbox")[1].check().run()
    assert at.get("markdown")[1].value == "Cleared!"
    messages_json = at.get("text")[-1].value
    assert messages_json == "[]"

    # Use message setter
    at.get("checkbox")[1].uncheck()
    at.get("checkbox")[2].check().run()
    messages_json = at.get("text")[-1].value
    assert "A basic message" in messages_json
    assert "extra cool message" in messages_json
