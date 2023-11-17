"""Base interface for loading large language model APIs."""
import json
from pathlib import Path
from typing import Union

import yaml

from langchain.llms import get_type_to_cls_dict
from langchain.llms.base import BaseLLM


def load_llm_from_config(config: dict, **llm_kwargs) -> BaseLLM:
    """Load LLM from Config Dict. Allow extra paramters with the kwargs"""
    if "_type" not in config:
        raise ValueError("Must specify an LLM Type in config")
    config_type = config.pop("_type")

    type_to_cls_dict = get_type_to_cls_dict()

    if config_type not in type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")
    extras = llm_kwargs.get(config_type, {})
    llm_cls = type_to_cls_dict[config_type]()
    return llm_cls(**config, **extras)


def load_llm(file: Union[str, Path], **llm_kwargs) -> BaseLLM:
    """Load LLM from file."""
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
    return load_llm_from_config(config, **llm_kwargs)
