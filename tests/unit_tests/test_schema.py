import pytest

from langchain.prompt import Prompt


def test_prompt_valid():
    template = "This is a {foo} test."
    input_variables = ["foo"]
    prompt = Prompt(input_variables=input_variables, template=template)
    assert prompt.template == template
    assert prompt.input_variables == input_variables


def test_prompt_missing_input_variables():
    template = "This is a {foo} test."
    input_variables = []
    with pytest.raises(ValueError):
        Prompt(input_variables=input_variables, template=template)


def test_prompt_extra_input_variables():
    template = "This is a {foo} test."
    input_variables = ["foo", "bar"]
    with pytest.raises(ValueError):
        Prompt(input_variables=input_variables, template=template)


def test_prompt_wrong_input_variables():
    template = "This is a {foo} test."
    input_variables = ["bar"]
    with pytest.raises(ValueError):
        Prompt(input_variables=input_variables, template=template)
