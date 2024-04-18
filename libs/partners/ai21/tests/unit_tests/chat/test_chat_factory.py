from typing import Type

import pytest

from langchain_ai21.chat.chat import (
    Chat,
    J2Chat,
    JambaChatCompletions,
)
from langchain_ai21.chat.chat_factory import create_chat
from tests.unit_tests.conftest import J2_CHAT_MODEL_NAME, JAMBA_CHAT_MODEL_NAME


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
    ],
    argnames=["model", "expected_chat_type"],
    argvalues=[
        (J2_CHAT_MODEL_NAME, J2Chat),
        (JAMBA_CHAT_MODEL_NAME, JambaChatCompletions),
    ],
)
def test_create_chat_with_supported_models(
        model: str,
        expected_chat_type: Type[Chat]) -> None:
    builder = create_chat(model)
    assert isinstance(builder, expected_chat_type)


def test_create_chat_with_supported_models__when_model_not_supported() -> None:
    with pytest.raises(ValueError):
        create_chat("unsupported-model")
