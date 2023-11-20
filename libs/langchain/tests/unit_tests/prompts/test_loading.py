"""Test loading functionality."""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from langchain.output_parsers import RegexParser
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.loading import load_prompt
from langchain.prompts.prompt import PromptTemplate

EXAMPLE_DIR = Path("tests/unit_tests/examples").absolute()


@contextmanager
def change_directory(dir: Path) -> Iterator:
    """Change the working directory to the right folder."""
    origin = Path().absolute()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(origin)


def test_loading_from_YAML() -> None:
    """Test loading from yaml file."""
    prompt = load_prompt(EXAMPLE_DIR / "simple_prompt.yaml")
    expected_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}.",
    )
    assert prompt == expected_prompt


def test_loading_from_JSON() -> None:
    """Test loading from json file."""
    prompt = load_prompt(EXAMPLE_DIR / "simple_prompt.json")
    expected_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}.",
    )
    assert prompt == expected_prompt


def test_loading_jinja_from_JSON() -> None:
    """Test that loading jinja2 format prompts from JSON raises ValueError."""
    prompt_path = EXAMPLE_DIR / "jinja_injection_prompt.json"
    with pytest.raises(ValueError, match=".*can lead to arbitrary code execution.*"):
        load_prompt(prompt_path)


def test_loading_jinja_from_YAML() -> None:
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


def test_loading_with_output_parser() -> None:
    with change_directory(EXAMPLE_DIR):
        prompt = load_prompt("prompt_with_output_parser.json")
        expected_template = "Given the following question and student answer, provide a correct answer and score the student answer.\nQuestion: {question}\nStudent Answer: {student_answer}\nCorrect Answer:"  # noqa: E501
        expected_prompt = PromptTemplate(
            input_variables=["question", "student_answer"],
            output_parser=RegexParser(
                regex="(.*?)\nScore: (.*)",
                output_keys=["answer", "score"],
            ),
            template=expected_template,
        )
        assert prompt == expected_prompt
