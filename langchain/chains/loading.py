"""Functionality for loading chains."""
import json
from pathlib import Path
from typing import Union

import yaml

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.loading import load_llm, load_llm_from_config
from langchain.prompts.loading import load_prompt, load_prompt_from_config


def _load_llm_chain(config: dict) -> LLMChain:
    """Load LLM chain from config dict."""
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"))
    else:
        raise ValueError("One of `llm` or `llm_config` must be present.")

    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    else:
        raise ValueError("One of `prompt` or `prompt_path` must be present.")

    return LLMChain(llm=llm, prompt=prompt, **config)


type_to_loader_dict = {"llm_chain": _load_llm_chain}


def load_chain_from_config(config: dict) -> Chain:
    """Load chain from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify an chain Type in config")
    config_type = config.pop("_type")

    if config_type not in type_to_loader_dict:
        raise ValueError(f"Loading {config_type} chain not supported")

    chain_loader = type_to_loader_dict[config_type]
    return chain_loader(config)


def load_chain(file: Union[str, Path]) -> Chain:
    """Load chain from file."""
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
    # Load the chain from the config now.
    return load_chain_from_config(config)
