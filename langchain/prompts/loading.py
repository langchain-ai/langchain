"""Load prompts from disk."""
import importlib
import json
import tempfile
from pathlib import Path
from typing import Union

import requests
import yaml

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


def load_prompt_from_config(config: dict) -> BasePromptTemplate:
    """Get the right type from the config and load it accordingly."""
    prompt_type = config.pop("_type", "prompt")
    if prompt_type == "prompt":
        return _load_prompt(config)
    elif prompt_type == "few_shot":
        return _load_few_shot_prompt(config)
    else:
        raise ValueError


def _load_template(var_name: str, config: dict) -> dict:
    """Load template from disk if applicable."""
    # Check if template_path exists in config.
    if f"{var_name}_path" in config:
        # If it does, make sure template variable doesn't also exist.
        if var_name in config:
            raise ValueError(
                f"Both `{var_name}_path` and `{var_name}` cannot be provided."
            )
        # Pop the template path from the config.
        template_path = Path(config.pop(f"{var_name}_path"))
        # Load the template.
        if template_path.suffix == ".txt":
            with open(template_path) as f:
                template = f.read()
        else:
            raise ValueError
        # Set the template variable to the extracted variable.
        config[var_name] = template
    return config


def _load_examples(config: dict) -> dict:
    """Load examples if necessary."""
    if isinstance(config["examples"], list):
        pass
    elif isinstance(config["examples"], str):
        with open(config["examples"]) as f:
            examples = json.load(f)
        config["examples"] = examples
    else:
        raise ValueError
    return config


def _load_few_shot_prompt(config: dict) -> FewShotPromptTemplate:
    """Load the few shot prompt from the config."""
    # Load the suffix and prefix templates.
    config = _load_template("suffix", config)
    config = _load_template("prefix", config)
    # Load the example prompt.
    if "example_prompt_path" in config:
        if "example_prompt" in config:
            raise ValueError(
                "Only one of example_prompt and example_prompt_path should "
                "be specified."
            )
        config["example_prompt"] = load_prompt(config.pop("example_prompt_path"))
    else:
        config["example_prompt"] = _load_prompt(config["example_prompt"])
    # Load the examples.
    config = _load_examples(config)
    return FewShotPromptTemplate(**config)


def _load_prompt(config: dict) -> PromptTemplate:
    """Load the prompt template from config."""
    # Load the template from disk if necessary.
    config = _load_template("template", config)
    return PromptTemplate(**config)


def load_prompt(file: Union[str, Path]) -> BasePromptTemplate:
    """Load prompt from file."""
    # Convert file to Path object.
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    # Load from either json or yaml.
    if file_path.suffix == ".json":
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix == ".yaml":
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    elif file_path.suffix == ".py":
        spec = importlib.util.spec_from_loader(
            "prompt", loader=None, origin=str(file_path)
        )
        if spec is None:
            raise ValueError("could not load spec")
        helper = importlib.util.module_from_spec(spec)
        with open(file_path, "rb") as f:
            exec(f.read(), helper.__dict__)
        if not isinstance(helper.PROMPT, BasePromptTemplate):
            raise ValueError("Did not get object of type BasePromptTemplate.")
        return helper.PROMPT
    else:
        raise ValueError(f"Got unsupported file type {file_path.suffix}")
    # Load the prompt from the config now.
    return load_prompt_from_config(config)


URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/prompts/"


def load_from_hub(path: str) -> BasePromptTemplate:
    """Load prompt from hub."""
    suffix = path.split(".")[-1]
    if suffix not in {"py", "json", "yaml"}:
        raise ValueError("Unsupported file type.")
    full_url = URL_BASE + path
    r = requests.get(full_url)
    if r.status_code != 200:
        raise ValueError(f"Could not find file at {full_url}")
    with tempfile.TemporaryDirectory() as tmpdirname:
        file = tmpdirname + "/prompt." + suffix
        with open(file, "wb") as f:
            f.write(r.content)
        return load_prompt(file)
