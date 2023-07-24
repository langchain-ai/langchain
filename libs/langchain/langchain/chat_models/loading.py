import json
from pathlib import Path
from typing import Union

import yaml

from langchain.chat_models import type_to_cls_dict
from langchain.chat_models.base import BaseChatModel


def load_chat_model_from_config(config: dict) -> BaseChatModel:
    """Load Chat Model from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify a Chat Model Type in config")
    config_type = config["_type"]

    if config_type not in type_to_cls_dict:
        raise ValueError(f"Loading {config_type} Chat Model not supported")

    llm_cls = type_to_cls_dict[config_type]
    del config["_type"]
    return llm_cls(**config)


def load_chat_model(file: Union[str, Path]) -> BaseChatModel:
    """Load Chat Model from file."""
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
    # Load the Chat Model from the config now.
    return load_chat_model_from_config(config)
