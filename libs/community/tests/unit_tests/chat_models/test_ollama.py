from typing import List, Literal, Optional

import pytest
from pydantic import BaseModel, ValidationError

from langchain_community.chat_models import ChatOllama


def test_standard_params() -> None:
    class ExpectedParams(BaseModel):
        ls_provider: str
        ls_model_name: str
        ls_model_type: Literal["chat", "llm"]
        ls_temperature: Optional[float]
        ls_max_tokens: Optional[int] = None
        ls_stop: Optional[List[str]] = None

    model = ChatOllama(model="llama3")
    ls_params = model._get_ls_params()
    try:
        ExpectedParams(**ls_params)
    except ValidationError as e:
        pytest.fail(f"Validation error: {e}")
    assert ls_params["ls_model_name"] == "llama3"

    # Test optional params
    model = ChatOllama(num_predict=10, stop=["test"], temperature=0.33)
    ls_params = model._get_ls_params()
    try:
        ExpectedParams(**ls_params)
    except ValidationError as e:
        pytest.fail(f"Validation error: {e}")
    assert ls_params["ls_max_tokens"] == 10
    assert ls_params["ls_stop"] == ["test"]
    assert ls_params["ls_temperature"] == 0.33
