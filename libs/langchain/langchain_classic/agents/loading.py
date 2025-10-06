"""Functionality for loading agents."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool

from langchain_classic.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_classic.agents.types import AGENT_TO_CLASS
from langchain_classic.chains.loading import load_chain, load_chain_from_config

logger = logging.getLogger(__name__)

URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/agents/"


def _load_agent_from_tools(
    config: dict,
    llm: BaseLanguageModel,
    tools: list[Tool],
    **kwargs: Any,
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    config_type = config.pop("_type")
    if config_type not in AGENT_TO_CLASS:
        msg = f"Loading {config_type} agent not supported"
        raise ValueError(msg)

    agent_cls = AGENT_TO_CLASS[config_type]
    combined_config = {**config, **kwargs}
    return agent_cls.from_llm_and_tools(llm, tools, **combined_config)


@deprecated("0.1.0", removal="1.0")
def load_agent_from_config(
    config: dict,
    llm: BaseLanguageModel | None = None,
    tools: list[Tool] | None = None,
    **kwargs: Any,
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    """Load agent from Config Dict.

    Args:
        config: Config dict to load agent from.
        llm: Language model to use as the agent.
        tools: List of tools this agent has access to.
        kwargs: Additional keyword arguments passed to the agent executor.

    Returns:
        An agent executor.

    Raises:
        ValueError: If agent type is not specified in the config.
    """
    if "_type" not in config:
        msg = "Must specify an agent Type in config"
        raise ValueError(msg)
    load_from_tools = config.pop("load_from_llm_and_tools", False)
    if load_from_tools:
        if llm is None:
            msg = (
                "If `load_from_llm_and_tools` is set to True, then LLM must be provided"
            )
            raise ValueError(msg)
        if tools is None:
            msg = (
                "If `load_from_llm_and_tools` is set to True, "
                "then tools must be provided"
            )
            raise ValueError(msg)
        return _load_agent_from_tools(config, llm, tools, **kwargs)
    config_type = config.pop("_type")

    if config_type not in AGENT_TO_CLASS:
        msg = f"Loading {config_type} agent not supported"
        raise ValueError(msg)

    agent_cls = AGENT_TO_CLASS[config_type]
    if "llm_chain" in config:
        config["llm_chain"] = load_chain_from_config(config.pop("llm_chain"))
    elif "llm_chain_path" in config:
        config["llm_chain"] = load_chain(config.pop("llm_chain_path"))
    else:
        msg = "One of `llm_chain` and `llm_chain_path` should be specified."
        raise ValueError(msg)
    if "output_parser" in config:
        logger.warning(
            "Currently loading output parsers on agent is not supported, "
            "will just use the default one.",
        )
        del config["output_parser"]

    combined_config = {**config, **kwargs}
    return agent_cls(**combined_config)


@deprecated("0.1.0", removal="1.0")
def load_agent(
    path: str | Path,
    **kwargs: Any,
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    """Unified method for loading an agent from LangChainHub or local fs.

    Args:
        path: Path to the agent file.
        kwargs: Additional keyword arguments passed to the agent executor.

    Returns:
        An agent executor.

    Raises:
        RuntimeError: If loading from the deprecated github-based
            Hub is attempted.
    """
    if isinstance(path, str) and path.startswith("lc://"):
        msg = (
            "Loading from the deprecated github-based Hub is no longer supported. "
            "Please use the new LangChain Hub at https://smith.langchain.com/hub "
            "instead."
        )
        raise RuntimeError(msg)
    return _load_agent_from_file(path, **kwargs)


def _load_agent_from_file(
    file: str | Path,
    **kwargs: Any,
) -> BaseSingleActionAgent | BaseMultiActionAgent:
    """Load agent from file."""
    valid_suffixes = {"json", "yaml"}
    # Convert file to Path object.
    file_path = Path(file) if isinstance(file, str) else file
    # Load from either json or yaml.
    if file_path.suffix[1:] == "json":
        with file_path.open() as f:
            config = json.load(f)
    elif file_path.suffix[1:] == "yaml":
        with file_path.open() as f:
            config = yaml.safe_load(f)
    else:
        msg = f"Unsupported file type, must be one of {valid_suffixes}."
        raise ValueError(msg)
    # Load the agent from the config now.
    return load_agent_from_config(config, **kwargs)
