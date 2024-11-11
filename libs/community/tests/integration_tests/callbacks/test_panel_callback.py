"""Integration tests for the PanelCallbackHandler module."""

import pytest

# Import the internal PanelCallbackHandler from its module - and not from
# the `langchain_community.callbacks.streamlit` package - so that we don't end up using
# Streamlit's externally-provided callback handler.
from langchain_community.callbacks.panel_callback import (
    PanelCallbackHandler,
)
from langchain_community.llms import OpenAI


@pytest.mark.requires("panel")
def test_panel_callback() -> None:
    import panel as pn

    chat_interface = pn.chat.ChatInterface()
    callback = PanelCallbackHandler(chat_interface)
    llm = OpenAI(temperature=0, streaming=True, callbacks=[callback])
    llm.invoke("hey")
    objects = chat_interface.objects
    assert len(objects) == 1
    assert objects[0].avatar == pn.chat.message.DEFAULT_AVATARS["langchain"]
    assert objects[0].user == "LangChain (gpt-4o-mini)"
