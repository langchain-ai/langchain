import os

import pytest

from langchain_community.llms.moonshot import Moonshot

os.environ["MOONSHOT_API_KEY"] = "key"


@pytest.mark.requires("openai")
def test_moonshot_model_param() -> None:
    llm = Moonshot(model="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    llm = Moonshot(model_name="bar")  # type: ignore[call-arg]
    assert llm.model_name == "bar"
