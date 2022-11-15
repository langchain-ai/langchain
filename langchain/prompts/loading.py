from pathlib import Path
from typing import Union
import yaml

from langchain.prompts import Prompt, DynamicPrompt
import json


def load_prompt_from_config(config):
    """Get the right type from the config and load it accordingly."""
    if "type" in config:
        prompt_type = config.pop("type")
    else:
        # Default to base prompt type.
        prompt_type = "prompt"
    if prompt_type == "prompt":
        return _load_prompt(config)
    elif prompt_type == "dynamic_prompt":
        return _load_dynamic_prompt(config)
    else:
        raise ValueError


def _load_template(var_name: str, config: dict) -> dict:
    """Load template from disk if applicable."""
    # Check if template_path exists in config.
    if f"{var_name}_path" in config:
        # If it does, make sure template variable doesn't also exist.
        if var_name in config:
            raise ValueError(f"Both `{var_name}_path` and `{var_name}` cannot be provided.")
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


def _load_examples(config):
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


def _load_dynamic_prompt(config):
    """Load the dynamic prompt from the config."""
    # Get the loader type (init, from_examples, etc)
    if "loader" in config:
        prompt_type = config.pop("loader")
    else:
        prompt_type = "init"
    # Call loading logic depending on what loader to use.
    if prompt_type == "init":
        # Load the suffix and prefix templates.
        config = _load_template("suffix", config)
        config = _load_template("prefix", config)
        return DynamicPrompt(**config)
    elif prompt_type == "from_structured_examples":
        # Load the suffix and prefix templates.
        config = _load_template("suffix", config)
        config = _load_template("prefix", config)
        # Load the example prompt.
        config["example_prompt"] = _load_prompt(config["example_prompt"])
        # Load the examples.
        config = _load_examples(config)
        return DynamicPrompt.from_structured_examples(**config)
    else:
        raise ValueError


def _load_prompt(config):
    """Load the base prompt type from config."""
    # Get the loader type (init, from_examples, etc)
    if "loader" in config:
        prompt_type = config.pop("loader")
    else:
        prompt_type = "init"
    # Call loading logic depending on what loader to use.
    if prompt_type == "init":
        # Load the template from disk.
        config = _load_template("template", config)
        return Prompt(**config)
    elif prompt_type == "from_examples":
        # Load the suffix and prefix templates.
        config = _load_template("suffix", config)
        config = _load_template("prefix", config)
        # Load the examples.
        config = _load_examples(config)
        return Prompt.from_examples(**config)
    elif prompt_type == "from_structured_examples":
        # Load the suffix and prefix templates.
        config = _load_template("suffix", config)
        config = _load_template("prefix", config)
        config["example_prompt"] = _load_prompt(config["example_prompt"])
        # Load the examples.
        config = _load_examples(config)
        return Prompt.from_structured_examples(**config)
    else:
        raise ValueError


def load_prompt(file: Union[str, Path]):
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
    else:
        raise ValueError
    # Load the prompt from the config now.
    return load_prompt_from_config(config)