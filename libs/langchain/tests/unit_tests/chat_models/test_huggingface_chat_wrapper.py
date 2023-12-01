"""Test Hugging Face Chat Wrapper."""
from importlib import import_module


def test_import_class():
    """Test that the class can be imported."""
    module_name = "langchain.chat_models.huggingface_chat_wrapper"
    class_name = "HuggingFaceChatWrapper"

    module = import_module(module_name)
    assert hasattr(module, class_name)
