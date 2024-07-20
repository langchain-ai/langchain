from typing import Type

import pytest

from langchain_ai21.chat.chat_adapter import (
    ChatAdapter,
    J2ChatAdapter,
    JambaChatCompletionsAdapter,
)
from langchain_ai21.chat.chat_factory import create_chat_adapter
from tests.unit_tests.conftest import J2_CHAT_MODEL_NAME, JAMBA_CHAT_MODEL_NAME


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
    ],
    argnames=["model", "expected_chat_type"],
    argvalues=[
        (J2_CHAT_MODEL_NAME, J2ChatAdapter),
        (JAMBA_CHAT_MODEL_NAME, JambaChatCompletionsAdapter),
    ],
)
def test_create_chat_adapter_with_supported_models(
    model: str, expected_chat_type: Type[ChatAdapter]
) -> None:
    adapter = create_chat_adapter(model)
    assert isinstance(adapter, expected_chat_type)


def test_create_chat_adapter__when_model_not_supported() -> None:
    with pytest.raises(ValueError):
        create_chat_adapter("unsupported-model")
