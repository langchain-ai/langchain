"""Test functionality related to prompts."""
import pytest

from langchain.prompt import Prompt


def test_prompt_valid() -> None:
    """Test prompts can be constructed."""
    template = "This is a {foo} test."
    input_variables = ["foo"]
    prompt = Prompt(input_variables=input_variables, template=template)
    assert prompt.template == template
    assert prompt.input_variables == input_variables


def test_prompt_missing_input_variables() -> None:
    """Test error is raised when input variables are not provided."""
    template = "This is a {foo} test."
    input_variables: list = []
    with pytest.raises(ValueError):
        Prompt(input_variables=input_variables, template=template)


def test_prompt_extra_input_variables() -> None:
    """Test error is raised when there are too many input variables."""
    template = "This is a {foo} test."
    input_variables = ["foo", "bar"]
    with pytest.raises(ValueError):
        Prompt(input_variables=input_variables, template=template)


def test_prompt_wrong_input_variables() -> None:
    """Test error is raised when name of input variable is wrong."""
    template = "This is a {foo} test."
    input_variables = ["bar"]
    with pytest.raises(ValueError):
        Prompt(input_variables=input_variables, template=template)


def test_prompt_invalid_template_format() -> None:
    """Test initializing a prompt with invalid template format."""
    template = "This is a {foo} test."
    input_variables = ["foo"]
    with pytest.raises(ValueError):
        Prompt(
            input_variables=input_variables, template=template, template_format="bar"
        )
