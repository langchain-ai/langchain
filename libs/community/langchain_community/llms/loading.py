"""Base interface for loading large language model APIs."""
import json
from pathlib import Path
from typing import Any, Union

import yaml
from langchain_core.language_models.llms import BaseLLM

from langchain_community.llms import get_type_to_cls_dict

_ALLOW_DANGEROUS_DESERIALIZATION_ARG = "allow_dangerous_deserialization"


def load_llm_from_config(config: dict, **kwargs: Any) -> BaseLLM:
    """Load LLM from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify an LLM Type in config")
    config_type = config.pop("_type")

    type_to_cls_dict = get_type_to_cls_dict()

    if config_type not in type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")

    llm_cls = type_to_cls_dict[config_type]()

    load_kwargs = {}
    if _ALLOW_DANGEROUS_DESERIALIZATION_ARG in llm_cls.__fields__:
        load_kwargs[_ALLOW_DANGEROUS_DESERIALIZATION_ARG] = kwargs.get(
            _ALLOW_DANGEROUS_DESERIALIZATION_ARG, False
        )

    return llm_cls(**config, **load_kwargs)


def load_llm(file: Union[str, Path], **kwargs: Any) -> BaseLLM:
    # Convert file to Path object.
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    # Load from either json or yaml.
    if file_path.suffix == ".json":
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("File type must be json or yaml")
    # Load the LLM from the config now.
    return load_llm_from_config(config, **kwargs)
