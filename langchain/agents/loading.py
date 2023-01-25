"""Functionality for loading agents."""
import json
from pathlib import Path
from typing import Union

import yaml

from langchain.agents.agent import Agent
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from la
from langchain.chains.loading import load_chain, load_chain_from_config




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
