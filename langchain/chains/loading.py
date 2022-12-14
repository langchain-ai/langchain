"""Base interface for loading chains."""
import json
from pathlib import Path
from typing import Union

import yaml

from langchain.chains import type_to_cls_dict
from langchain.chains.base import Chain


def load_chain_from_config(config: dict) -> Chain:
    """Load LLM from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify an LLM Type in config")
    config_type = config.pop("_type")
    if config_type not in type_to_cls_dict:
        raise ValueError(f"Loading {config_type} Chains not current supported.")
    llm_cls = type_to_cls_dict[config_type]
    return llm_cls.from_config(config)


def load_chain(file: Union[str, Path]) -> Chain:
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
    return load_chain_from_config(config)
