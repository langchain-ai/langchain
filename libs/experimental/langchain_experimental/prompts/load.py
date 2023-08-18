# Susceptible to arbitrary code execution: https://github.com/hwchase17/langchain/issues/4849
import importlib
import json
from pathlib import Path
from typing import Union

import yaml
from langchain.prompts.loading import load_prompt_from_config, try_load_from_hub
from langchain.schema.prompts import BasePromptTemplate


def load_prompt(path: Union[str, Path]) -> BasePromptTemplate:
    """Unified method for loading a prompt from LangChainHub or local fs."""
    if hub_result := try_load_from_hub(
        path, _load_prompt_from_file, "prompts", {"py", "json", "yaml"}
    ):
        return hub_result
    else:
        return _load_prompt_from_file(path)


def _load_prompt_from_file(file: Union[str, Path]) -> BasePromptTemplate:
    """Load prompt from file."""
    # Convert file to a Path object.
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
