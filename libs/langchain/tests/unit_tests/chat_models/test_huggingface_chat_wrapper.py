"""Test Hugging Face Chat Wrapper."""


def test_import_class():
    try:
        from langchain.chat_models import HuggingFaceChatWrapper
    except ImportError:
        assert False, "Failed to import HuggingFaceChatWrapper"
