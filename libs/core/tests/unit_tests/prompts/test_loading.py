"""Test loading functionality."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.loading import load_prompt
from langchain_core.prompts.prompt import PromptTemplate

EXAMPLE_DIR = (Path(__file__).parent.parent / "examples").absolute()


@contextmanager
def change_directory(dir: Path) -> Iterator:
    """Change the working directory to the right folder."""
    origin = Path().absolute()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(origin)


def test_loading_from_yaml() -> None:
    """Test loading from yaml file."""
    prompt = load_prompt(EXAMPLE_DIR / "simple_prompt.yaml")
    expected_prompt = PromptTemplate(
        input_variables=["adjective"],
        partial_variables={"content": "dogs"},
        template="Tell me a {adjective} joke about {content}.",
    )
    assert prompt == expected_prompt


def test_loading_chat_from_yaml() -> None:
    """Test loading from yaml file."""
    prompt = load_prompt(EXAMPLE_DIR / "simple_chat_prompt.yaml")
    expected_prompt = ChatPromptTemplate(
        input_variables=["adjective"],
        partial_variables={"content": "dogs"},
        messages=[
            ("system", "You are a comedian"),
            ("human", "Tell me a {adjective} joke about {content}."),
        ],
    )
    assert prompt == expected_prompt


def test_loading_from_json() -> None:
    """Test loading from json file."""
    prompt = load_prompt(EXAMPLE_DIR / "simple_prompt.json")
    expected_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}.",
    )
    assert prompt == expected_prompt


def test_loading_chat_from_json() -> None:
    """Test loading from json file."""
    prompt = load_prompt(EXAMPLE_DIR / "simple_chat_prompt.json")
    expected_prompt = ChatPromptTemplate(
        input_variables=["adjective"],
        partial_variables={"content": "dogs"},
        messages=[
            ("system", "You are a comedian"),
            ("human", "Tell me a {adjective} joke about {content}."),
        ],
    )
    assert prompt == expected_prompt


def test_loading_jinja_from_json() -> None:
    """Test that loading jinja2 format prompts from JSON raises ValueError."""
    prompt_path = EXAMPLE_DIR / "jinja_injection_prompt.json"
    with pytest.raises(ValueError, match=".*can lead to arbitrary code execution.*"):
        load_prompt(prompt_path)


def test_loading_jinja_from_yaml() -> None:
    """Test that loading jinja2 format prompts from YAML raises ValueError."""
    prompt_path = EXAMPLE_DIR / "jinja_injection_prompt.yaml"
    with pytest.raises(ValueError, match=".*can lead to arbitrary code execution.*"):
        load_prompt(prompt_path)


def test_saving_loading_round_trip(tmp_path: Path) -> None:
    """Test equality when saving and loading a prompt."""
    simple_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}.",
    )
    simple_prompt.save(file_path=tmp_path / "prompt.yaml")
    loaded_prompt = load_prompt(tmp_path / "prompt.yaml")
    assert loaded_prompt == simple_prompt

    few_shot_prompt = FewShotPromptTemplate(
        input_variables=["adjective"],
        prefix="Write antonyms for the following words.",
        example_prompt=PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}",
        ),
        examples=[
            {"input": "happy", "output": "sad"},
            {"input": "tall", "output": "short"},
        ],
        suffix="Input: {adjective}\nOutput:",
    )
    few_shot_prompt.save(file_path=tmp_path / "few_shot.yaml")
    loaded_prompt = load_prompt(tmp_path / "few_shot.yaml")
    assert loaded_prompt == few_shot_prompt


def test_loading_with_template_as_file() -> None:
    """Test loading when the template is a file."""
    with change_directory(EXAMPLE_DIR):
        prompt = load_prompt("simple_prompt_with_template_file.json")
        expected_prompt = PromptTemplate(
            input_variables=["adjective", "content"],
            template="Tell me a {adjective} joke about {content}.",
        )
        assert prompt == expected_prompt


def test_loading_few_shot_prompt_from_yaml() -> None:
    """Test loading few shot prompt from yaml."""
    with change_directory(EXAMPLE_DIR):
        prompt = load_prompt("few_shot_prompt.yaml")
        expected_prompt = FewShotPromptTemplate(
            input_variables=["adjective"],
            prefix="Write antonyms for the following words.",
            example_prompt=PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}",
            ),
            examples=[
                {"input": "happy", "output": "sad"},
                {"input": "tall", "output": "short"},
            ],
            suffix="Input: {adjective}\nOutput:",
        )
        assert prompt == expected_prompt


def test_loading_few_shot_prompt_from_json() -> None:
    """Test loading few shot prompt from json."""
    with change_directory(EXAMPLE_DIR):
        prompt = load_prompt("few_shot_prompt.json")
        expected_prompt = FewShotPromptTemplate(
            input_variables=["adjective"],
            prefix="Write antonyms for the following words.",
            example_prompt=PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}",
            ),
            examples=[
                {"input": "happy", "output": "sad"},
                {"input": "tall", "output": "short"},
            ],
            suffix="Input: {adjective}\nOutput:",
        )
        assert prompt == expected_prompt


def test_loading_few_shot_prompt_when_examples_in_config() -> None:
    """Test loading few shot prompt when the examples are in the config."""
    with change_directory(EXAMPLE_DIR):
        prompt = load_prompt("few_shot_prompt_examples_in.json")
        expected_prompt = FewShotPromptTemplate(
            input_variables=["adjective"],
            prefix="Write antonyms for the following words.",
            example_prompt=PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}",
            ),
            examples=[
                {"input": "happy", "output": "sad"},
                {"input": "tall", "output": "short"},
            ],
            suffix="Input: {adjective}\nOutput:",
        )
        assert prompt == expected_prompt


def test_loading_few_shot_prompt_example_prompt() -> None:
    """Test loading few shot when the example prompt is in its own file."""
    with change_directory(EXAMPLE_DIR):
        prompt = load_prompt("few_shot_prompt_example_prompt.json")
        expected_prompt = FewShotPromptTemplate(
            input_variables=["adjective"],
            prefix="Write antonyms for the following words.",
            example_prompt=PromptTemplate(
                input_variables=["input", "output"],
                template="Input: {input}\nOutput: {output}",
            ),
            examples=[
                {"input": "happy", "output": "sad"},
                {"input": "tall", "output": "short"},
            ],
            suffix="Input: {adjective}\nOutput:",
        )
        assert prompt == expected_prompt
