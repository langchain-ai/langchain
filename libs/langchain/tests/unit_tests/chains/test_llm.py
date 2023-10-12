"""Test LLM chain."""
from tempfile import TemporaryDirectory
from typing import Dict, List, Union
from unittest.mock import patch

import pytest

from langchain.chains.llm import LLMChain
from langchain.chains.loading import load_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseOutputParser
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeOutputParser(BaseOutputParser):
    """Fake output parser class for testing."""

    def parse(self, text: str) -> Union[str, List[str], Dict[str, str]]:
        """Parse by splitting."""
        return text.split()


@pytest.fixture
def fake_llm_chain() -> LLMChain:
    """Fake LLM chain for testing purposes."""
    prompt = PromptTemplate(input_variables=["bar"], template="This is a {bar}:")
    return LLMChain(prompt=prompt, llm=FakeLLM(), output_key="text1")


@patch("langchain.llms.loading.get_type_to_cls_dict", lambda: {"fake": lambda: FakeLLM})
def test_serialization(fake_llm_chain: LLMChain) -> None:
    """Test serialization."""
    with TemporaryDirectory() as temp_dir:
        file = temp_dir + "/llm.json"
        fake_llm_chain.save(file)
        loaded_chain = load_chain(file)
        assert loaded_chain == fake_llm_chain


def test_missing_inputs(fake_llm_chain: LLMChain) -> None:
    """Test error is raised if inputs are missing."""
    with pytest.raises(ValueError):
        fake_llm_chain({"foo": "bar"})


def test_valid_call(fake_llm_chain: LLMChain) -> None:
    """Test valid call of LLM chain."""
    output = fake_llm_chain({"bar": "baz"})
    assert output == {"bar": "baz", "text1": "foo"}

    # Test with stop words.
    output = fake_llm_chain({"bar": "baz", "stop": ["foo"]})
    # Response should be `bar` now.
    assert output == {"bar": "baz", "stop": ["foo"], "text1": "bar"}


def test_predict_method(fake_llm_chain: LLMChain) -> None:
    """Test predict method works."""
    output = fake_llm_chain.predict(bar="baz")
    assert output == "foo"


def test_predict_and_parse() -> None:
    """Test parsing ability."""
    prompt = PromptTemplate(
        input_variables=["foo"], template="{foo}", output_parser=FakeOutputParser()
    )
    llm = FakeLLM(queries={"foo": "foo bar"})
    chain = LLMChain(prompt=prompt, llm=llm)
    output = chain.predict_and_parse(foo="foo")
    assert output == ["foo", "bar"]
