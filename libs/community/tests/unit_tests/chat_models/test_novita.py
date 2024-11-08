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
        api_key="787ee3fd-ff97-4aac-936e-1b09cf74a559",
        model="gryphe/mythomax-l2-13b",
    )
    assert chat.novita_api_key.get_secret_value() == "787ee3fd-ff97-4aac-936e-1b09cf74a559"
    assert chat.model_name == "gryphe/mythomax-l2-13b"
