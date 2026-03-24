"""Test loading functionality."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from langchain_core._api import suppress_langchain_deprecation_warning
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.loading import (
    _load_examples,
    _load_template,
    load_prompt,
    load_prompt_from_config,
)
from langchain_core.prompts.prompt import PromptTemplate

EXAMPLE_DIR = (Path(__file__).parent.parent / "examples").absolute()


@contextmanager
def change_directory(dir_path: Path) -> Iterator[None]:
    """Change the working directory to the right folder."""
    origin = Path().absolute()
    try:
        os.chdir(dir_path)
        yield
    finally:
        os.chdir(origin)


def test_loading_from_yaml() -> None:
    """Test loading from yaml file."""
    with suppress_langchain_deprecation_warning():
        prompt = load_prompt(EXAMPLE_DIR / "simple_prompt.yaml")
    expected_prompt = PromptTemplate(
        input_variables=["adjective"],
        partial_variables={"content": "dogs"},
        template="Tell me a {adjective} joke about {content}.",
    )
    assert prompt == expected_prompt


def test_loading_from_json() -> None:
    """Test loading from json file."""
    with suppress_langchain_deprecation_warning():
        prompt = load_prompt(EXAMPLE_DIR / "simple_prompt.json")
    expected_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}.",
    )
    assert prompt == expected_prompt


def test_loading_jinja_from_json() -> None:
    """Test that loading jinja2 format prompts from JSON raises ValueError."""
    prompt_path = EXAMPLE_DIR / "jinja_injection_prompt.json"
    with (
        suppress_langchain_deprecation_warning(),
        pytest.raises(ValueError, match=r".*can lead to arbitrary code execution.*"),
    ):
        load_prompt(prompt_path)


def test_loading_jinja_from_yaml() -> None:
    """Test that loading jinja2 format prompts from YAML raises ValueError."""
    prompt_path = EXAMPLE_DIR / "jinja_injection_prompt.yaml"
    with (
        suppress_langchain_deprecation_warning(),
        pytest.raises(ValueError, match=r".*can lead to arbitrary code execution.*"),
    ):
        load_prompt(prompt_path)


def test_saving_loading_round_trip(tmp_path: Path) -> None:
    """Test equality when saving and loading a prompt."""
    simple_prompt = PromptTemplate(
        input_variables=["adjective", "content"],
        template="Tell me a {adjective} joke about {content}.",
    )
    with suppress_langchain_deprecation_warning():
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
    with suppress_langchain_deprecation_warning():
        few_shot_prompt.save(file_path=tmp_path / "few_shot.yaml")
        loaded_prompt = load_prompt(tmp_path / "few_shot.yaml")
    assert loaded_prompt == few_shot_prompt


def test_loading_with_template_as_file() -> None:
    """Test loading when the template is a file."""
    with change_directory(EXAMPLE_DIR), suppress_langchain_deprecation_warning():
        prompt = load_prompt(
            "simple_prompt_with_template_file.json", allow_dangerous_paths=True
        )
        expected_prompt = PromptTemplate(
            input_variables=["adjective", "content"],
            template="Tell me a {adjective} joke about {content}.",
        )
        assert prompt == expected_prompt


def test_load_template_rejects_absolute_path(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET")
    config = {"template_path": str(secret)}
    with pytest.raises(ValueError, match="is absolute"):
        _load_template("template", config)


def test_load_template_rejects_traversal() -> None:
    config = {"template_path": "../../etc/secret.txt"}
    with pytest.raises(ValueError, match=r"contains '\.\.' components"):
        _load_template("template", config)


def test_load_template_allows_dangerous_paths_when_opted_in(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET")
    config = {"template_path": str(secret)}
    result = _load_template("template", config, allow_dangerous_paths=True)
    assert result["template"] == "SECRET"


def test_load_examples_rejects_absolute_path(tmp_path: Path) -> None:
    examples_file = tmp_path / "examples.json"
    examples_file.write_text(json.dumps([{"input": "a", "output": "b"}]))
    config = {"examples": str(examples_file)}
    with pytest.raises(ValueError, match="is absolute"):
        _load_examples(config)


def test_load_examples_rejects_traversal() -> None:
    config = {"examples": "../../secrets/data.json"}
    with pytest.raises(ValueError, match=r"contains '\.\.' components"):
        _load_examples(config)


def test_load_examples_allows_dangerous_paths_when_opted_in(tmp_path: Path) -> None:
    examples_file = tmp_path / "examples.json"
    examples_file.write_text(json.dumps([{"input": "a", "output": "b"}]))
    config = {"examples": str(examples_file)}
    result = _load_examples(config, allow_dangerous_paths=True)
    assert result["examples"] == [{"input": "a", "output": "b"}]


def test_load_prompt_from_config_rejects_absolute_template_path(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET")
    config = {
        "_type": "prompt",
        "template_path": str(secret),
        "input_variables": [],
    }
    with (
        suppress_langchain_deprecation_warning(),
        pytest.raises(ValueError, match="is absolute"),
    ):
        load_prompt_from_config(config)


def test_load_prompt_from_config_rejects_traversal_template_path() -> None:
    config = {
        "_type": "prompt",
        "template_path": "../../../tmp/secret.txt",
        "input_variables": [],
    }
    with (
        suppress_langchain_deprecation_warning(),
        pytest.raises(ValueError, match=r"contains '\.\.' components"),
    ):
        load_prompt_from_config(config)


def test_load_prompt_from_config_allows_dangerous_paths(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET")
    config = {
        "_type": "prompt",
        "template_path": str(secret),
        "input_variables": [],
    }
    with suppress_langchain_deprecation_warning():
        prompt = load_prompt_from_config(config, allow_dangerous_paths=True)
    assert isinstance(prompt, PromptTemplate)
    assert prompt.template == "SECRET"


def test_load_prompt_from_config_few_shot_rejects_traversal_examples() -> None:
    config = {
        "_type": "few_shot",
        "input_variables": ["query"],
        "prefix": "Examples:",
        "example_prompt": {
            "_type": "prompt",
            "input_variables": ["input", "output"],
            "template": "{input}: {output}",
        },
        "examples": "../../../../.docker/config.json",
        "suffix": "Query: {query}",
    }
    with (
        suppress_langchain_deprecation_warning(),
        pytest.raises(ValueError, match=r"contains '\.\.' components"),
    ):
        load_prompt_from_config(config)


def test_load_prompt_from_config_few_shot_rejects_absolute_examples(
    tmp_path: Path,
) -> None:
    examples_file = tmp_path / "examples.json"
    examples_file.write_text(json.dumps([{"input": "a", "output": "b"}]))
    config = {
        "_type": "few_shot",
        "input_variables": ["query"],
        "prefix": "Examples:",
        "example_prompt": {
            "_type": "prompt",
            "input_variables": ["input", "output"],
            "template": "{input}: {output}",
        },
        "examples": str(examples_file),
        "suffix": "Query: {query}",
    }
    with (
        suppress_langchain_deprecation_warning(),
        pytest.raises(ValueError, match="is absolute"),
    ):
        load_prompt_from_config(config)


def test_load_prompt_from_config_few_shot_rejects_absolute_example_prompt_path(
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text(
        json.dumps(
            {
                "_type": "prompt",
                "template": "{input}: {output}",
                "input_variables": ["input", "output"],
            }
        )
    )
    config = {
        "_type": "few_shot",
        "input_variables": ["query"],
        "prefix": "Examples:",
        "example_prompt_path": str(prompt_file),
        "examples": [{"input": "a", "output": "b"}],
        "suffix": "Query: {query}",
    }
    with (
        suppress_langchain_deprecation_warning(),
        pytest.raises(ValueError, match="is absolute"),
    ):
        load_prompt_from_config(config)


def test_loading_few_shot_prompt_from_yaml() -> None:
    """Test loading few shot prompt from yaml."""
    with change_directory(EXAMPLE_DIR), suppress_langchain_deprecation_warning():
        prompt = load_prompt("few_shot_prompt.yaml", allow_dangerous_paths=True)
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
    with change_directory(EXAMPLE_DIR), suppress_langchain_deprecation_warning():
        prompt = load_prompt("few_shot_prompt.json", allow_dangerous_paths=True)
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
    with change_directory(EXAMPLE_DIR), suppress_langchain_deprecation_warning():
        prompt = load_prompt(
            "few_shot_prompt_examples_in.json", allow_dangerous_paths=True
        )
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
    with change_directory(EXAMPLE_DIR), suppress_langchain_deprecation_warning():
        prompt = load_prompt(
            "few_shot_prompt_example_prompt.json", allow_dangerous_paths=True
        )
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
