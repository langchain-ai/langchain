"""Test functionality related to prompts."""

from typing import Any

import pytest
from pydantic.v1 import ValidationError

from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts.injector import PromptInjector
from langchain_core.pydantic_v1 import BaseModel


class PydanticWithPrompt(BaseModel):
    prompt: Any
    other_attribute: Any


class PydanticWithoutPrompt(BaseModel):
    other_attribute: Any


def test_prompt_inject_to_object_with_attribute() -> None:
    """Test prompt can be injected to the objects come with 'prompt' attribute"""
    prompt = StringPromptValue(text="Dummy prompt")
    inject_object = PydanticWithPrompt()
    injector = PromptInjector([inject_object])
    output_of_injector = injector.invoke(prompt)
    assert output_of_injector is prompt
    assert inject_object.prompt is prompt


def test_prompt_inject_to_object_without_attribute_rejection() -> None:
    """Test prompt can't be injected to the objects
    without 'prompt' attribute"""
    inject_object = PydanticWithoutPrompt()
    with pytest.raises(ValidationError):
        PromptInjector([inject_object])


def test_prompt_inject_to_object_without_attribute_skip() -> None:
    """Test prompt injection to the objects skipped
    without 'prompt' attribute once pass_on_injection_fail is set to True"""
    prompt = StringPromptValue(text="Dummy prompt")
    inject_objects = [PydanticWithPrompt(), PydanticWithoutPrompt()]
    injector = PromptInjector(inject_objects, pass_on_injection_fail=True)
    output_of_injector = injector.invoke(prompt)
    assert output_of_injector is prompt
    assert inject_objects[0].prompt is prompt
    with pytest.raises(AttributeError, match="'PydanticWithoutPrompt' object has no attribute 'prompt'"):
        temp = inject_objects[1].prompt

