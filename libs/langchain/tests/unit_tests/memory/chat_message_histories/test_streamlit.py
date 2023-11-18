"""Unit tests for StreamlitChatMessageHistory functionality."""
import pytest

test_script = """
    import json
    import streamlit as st
    from langchain.memory import ConversationBufferMemory
    from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
    from langchain.schema.messages import _message_to_dict

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

    # Write the output to st.code as a json blob for inspection
    messages = memory.chat_memory.messages
    messages_json = json.dumps([_message_to_dict(msg) for msg in messages])
    st.text(messages_json)
"""


@pytest.mark.requires("streamlit")
def test_memory_with_message_store() -> None:
    try:
        from streamlit.testing.script_interactions import InteractiveScriptTests
    except ModuleNotFoundError:
        pytest.skip("Incorrect version of Streamlit installed")

    test_handler = InteractiveScriptTests()
    test_handler.setUp()
    try:
        sr = test_handler.script_from_string(test_script).run()
    except TypeError:
        # Earlier version expected 2 arguments
        sr = test_handler.script_from_string("memory_test.py", test_script).run()

    # Initial run should write two messages
    messages_json = sr.get("text")[-1].value
    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # Uncheck the initial write, they should persist in session_state
    sr = sr.get("checkbox")[0].uncheck().run()
    assert sr.get("markdown")[0].value == "Skipped add"
    messages_json = sr.get("text")[-1].value
    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # Clear the message history
    sr = sr.get("checkbox")[1].check().run()
    assert sr.get("markdown")[1].value == "Cleared!"
    messages_json = sr.get("text")[-1].value
    assert messages_json == "[]"
