from unittest import mock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import ValidationError

from langchain.chat_models.llamacpp import ChatLlamacpp


@pytest.mark.requires("llama-cpp-python")
def test_model_failed_init_from_path() -> None:
    with pytest.raises(ValidationError):
        ChatLlamacpp(model_path="./fake/path/to/model.ggml")


@pytest.mark.requires("llama-cpp-python")
@mock.patch("llama_cpp.Llama")
def test_model_init_from_path(model_mock: mock.MagicMock) -> None:
    model = ChatLlamacpp(model_path="./fake/path/to/model.ggml")

    model_mock.assert_called_once()
    assert isinstance(model, ChatLlamacpp)
    assert model.model_path == "./fake/path/to/model.ggml"


@pytest.mark.requires("llama-cpp-python")
@mock.patch("llama_cpp.Llama")
def test__format_msg(model_mock: mock.MagicMock) -> None:
    model = ChatLlamacpp(model_path="./fake/path/to/model.ggml")
    message_original = "This is a message"
    message_role = "message_role"

    formatted_chat_msg = f"\n\n{message_role.capitalize()}: {message_original}"
    model_mock.assert_called_once()

    assert f"<<SYS>> {message_original} <</SYS>>" == model._format_message_as_text(
        SystemMessage(content=message_original)
    )
    assert formatted_chat_msg == model._format_message_as_text(
        ChatMessage(content=message_original, role=message_role)
    )
    assert f"[INST] {message_original} [/INST]" == model._format_message_as_text(
        HumanMessage(content=message_original)
    )
    assert f"{message_original}" == model._format_message_as_text(
        AIMessage(content=message_original)
    )
