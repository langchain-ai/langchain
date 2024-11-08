"""Test Novita AI chat model"""

import pytest
from pydantic import ValidationError

from langchain_community.chat_models import ChatNovita


@pytest.mark.requires("openai")
def test__missing_novita_api_key() -> None:
    with pytest.raises(ValidationError) as e:
        ChatNovita()
    assert "Did not find novita_api_key" in str(e)


@pytest.mark.requires("openai")
def test__all_fields_provided() -> None:
    chat = ChatNovita(
        api_key="your_api_key",
        model="gryphe/mythomax-l2-13b",
    )
    assert chat.novita_api_key.get_secret_value() == "your_api_key"
    assert chat.model_name == "gryphe/mythomax-l2-13b"
