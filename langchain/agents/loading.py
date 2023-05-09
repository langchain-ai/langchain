"""Functionality for loading agents."""
import json
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml

from langchain.agents.agent import BaseSingleActionAgent
from langchain.agents.tools import Tool
from langchain.agents.types import AGENT_TO_CLASS
from langchain.base_language import BaseLanguageModel
from langchain.chains.loading import load_chain, load_chain_from_config
from langchain.utilities.loading import try_load_from_hub

URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/agents/"


def _load_agent_from_tools(
    config: dict, llm: BaseLanguageModel, tools: List[Tool], **kwargs: Any
) -> BaseSingleActionAgent:
    config_type = config.pop("_type")
    if config_type not in AGENT_TO_CLASS:
        raise ValueError(f"Loading {config_type} agent not supported")

    agent_cls = AGENT_TO_CLASS[config_type]
    combined_config = {**config, **kwargs}
    return agent_cls.from_llm_and_tools(llm, tools, **combined_config)


def load_agent_from_config(
    config: dict,
    llm: Optional[BaseLanguageModel] = None,
    tools: Optional[List[Tool]] = None,
    **kwargs: Any,
) -> BaseSingleActionAgent:
    """Load agent from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify an agent Type in config")
    load_from_tools = config.pop("load_from_llm_and_tools", False)
    if load_from_tools:
        if llm is None:
            raise ValueError(
                "If `load_from_llm_and_tools` is set to True, "
                "then LLM must be provided"
            )
        if tools is None:
            raise ValueError(
                "If `load_from_llm_and_tools` is set to True, "
                "then tools must be provided"
            )
        return _load_agent_from_tools(config, llm, tools, **kwargs)
    config_type = config.pop("_type")

    if config_type not in AGENT_TO_CLASS:
        raise ValueError(f"Loading {config_type} agent not supported")

    agent_cls = AGENT_TO_CLASS[config_type]
    if "llm_chain" in config:
        config["llm_chain"] = load_chain_from_config(config.pop("llm_chain"))
    elif "llm_chain_path" in config:
        config["llm_chain"] = load_chain(config.pop("llm_chain_path"))
    else:
        raise ValueError("One of `llm_chain` and `llm_chain_path` should be specified.")
    combined_config = {**config, **kwargs}
    return agent_cls(**combined_config)  # type: ignore


def load_agent(path: Union[str, Path], **kwargs: Any) -> BaseSingleActionAgent:
    """Unified method for loading a agent from LangChainHub or local fs."""
    if hub_result := try_load_from_hub(
        path, _load_agent_from_file, "agents", {"json", "yaml"}
    ):
        return hub_result
    else:
        return _load_agent_from_file(path, **kwargs)


def _load_agent_from_file(
    file: Union[str, Path], **kwargs: Any
) -> BaseSingleActionAgent:
    """Load agent from file."""
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
    # Load the agent from the config now.
    return load_agent_from_config(config, **kwargs)
