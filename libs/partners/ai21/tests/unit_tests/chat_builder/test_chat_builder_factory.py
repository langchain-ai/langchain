from typing import Type

import pytest

from langchain_ai21.chat_builder.chat_builder import (
    ChatBuilder,
    J2ChatBuilder,
    JambaChatBuilder,
)
from langchain_ai21.chat_builder.chat_builder_factory import create_chat_builder
from tests.unit_tests.conftest import J2_CHAT_MODEL_NAME, JAMBA_CHAT_MODEL_NAME


@pytest.mark.parametrize(
    ids=[
        "when_j2_model",
        "when_jamba_model",
    ],
    argnames=["model", "expected_builder_type"],
    argvalues=[
        (J2_CHAT_MODEL_NAME, J2ChatBuilder),
        (JAMBA_CHAT_MODEL_NAME, JambaChatBuilder),
    ],
)
def test_create_chat_builder_with_supported_models(
        model: str,
        expected_builder_type: Type[ChatBuilder]) -> None:
    builder = create_chat_builder(model)
    assert isinstance(builder, expected_builder_type)


def test_create_chat_builder_with_supported_models__when_model_not_supported() -> None:
    with pytest.raises(ValueError):
        create_chat_builder("unsupported-model")
