"""Functionality for loading agents."""
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml
from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.utils.loading import try_load_from_hub

from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.tools import Tool
from langchain.agents.types import AGENT_TO_CLASS
from langchain.chains.loading import load_chain, load_chain_from_config

logger = logging.getLogger(__file__)

URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/agents/"


def _load_agent_from_tools(
    config: dict, llm: BaseLanguageModel, tools: List[Tool], **kwargs: Any
) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
    config_type = config.pop("_type")
    if config_type not in AGENT_TO_CLASS:
        raise ValueError(f"Loading {config_type} agent not supported")

    agent_cls = AGENT_TO_CLASS[config_type]
    combined_config = {**config, **kwargs}
    return agent_cls.from_llm_and_tools(llm, tools, **combined_config)


@deprecated("0.1.0", removal="0.2.0")
def load_agent_from_config(
    config: dict,
    llm: Optional[BaseLanguageModel] = None,
    tools: Optional[List[Tool]] = None,
    **kwargs: Any,
) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
    """Load agent from Config Dict.

    Args:
        config: Config dict to load agent from.
        llm: Language model to use as the agent.
        tools: List of tools this agent has access to.
        **kwargs: Additional keyword arguments passed to the agent executor.

    Returns:
        An agent executor.
    """
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
    if "output_parser" in config:
        logger.warning(
            "Currently loading output parsers on agent is not supported, "
            "will just use the default one."
        )
        del config["output_parser"]

    combined_config = {**config, **kwargs}
    return agent_cls(**combined_config)  # type: ignore


@deprecated("0.1.0", removal="0.2.0")
def load_agent(
    path: Union[str, Path], **kwargs: Any
) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
    """Unified method for loading an agent from LangChainHub or local fs.

    Args:
        path: Path to the agent file.
        **kwargs: Additional keyword arguments passed to the agent executor.

    Returns:
        An agent executor.
    """
    valid_suffixes = {"json", "yaml"}
    if hub_result := try_load_from_hub(
        path, _load_agent_from_file, "agents", valid_suffixes
    ):
        return hub_result
    else:
        return _load_agent_from_file(path, **kwargs)


def _load_agent_from_file(
    file: Union[str, Path], **kwargs: Any
) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
    """Load agent from file."""
    valid_suffixes = {"json", "yaml"}
    # Convert file to Path object.
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    # Load from either json or yaml.
    if file_path.suffix[1:] == "json":
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix[1:] == "yaml":
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file type, must be one of {valid_suffixes}.")
    # Load the agent from the config now.
    return load_agent_from_config(config, **kwargs)
