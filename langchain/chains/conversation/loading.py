"""Base interface for loading memory."""
import json
from pathlib import Path
from typing import Union

import yaml

from langchain.chains.base import Memory
from langchain.chains.conversation.memory import type_to_cls_dict


def load_memory_from_config(config: dict) -> Memory:
    """Load Memory from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify an LLM Type in config")
    config_type = config.pop("_type")

    if config_type not in type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")

    memory_cls = type_to_cls_dict[config_type]
    return memory_cls(**config)


def load_memory(file: Union[str, Path]) -> Memory:
    """Load Memory from file."""
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
        raise ValueError("File type must be json or yaml")
    # Load the LLM from the config now.
    return load_memory_from_config(config)
