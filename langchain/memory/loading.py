"""Base interface for loading memory."""
import json
from pathlib import Path
from typing import Union

import yaml

from langchain.llms.loading import load_llm_from_config
from langchain.memory import type_to_cls_dict
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.prompts.loading import load_prompt_from_config
from langchain.schema import BaseMemory, _message_from_dict


def load_memory_from_config(config: dict) -> BaseMemory:
    """Load Memory from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify a Memory Type in config")
    config_type = config.pop("_type")

    if config_type not in type_to_cls_dict:
        raise ValueError(f"Loading {config_type} Memory not supported")

    # Load chat memory if it exists
    if "chat_memory" in config:
        chat_memory_config = config.pop("chat_memory")
        config["chat_memory"] = load_chat_memory_from_config(chat_memory_config)

    # Load LLM if it exists
    if "llm" in config:
        llm_config = config.pop("llm")
        config["llm"] = load_llm_from_config(llm_config)

    # Load prompt if it exists
    if "entity_extraction_prompt" in config:
        prompt_config = config.pop("entity_extraction_prompt")
        config["entity_extraction_prompt"] = load_prompt_from_config(prompt_config)
    if "entity_summarization_prompt" in config:
        prompt_config = config.pop("entity_summarization_prompt")
        config["entity_summarization_prompt"] = load_prompt_from_config(prompt_config)

    memory_cls = type_to_cls_dict[config_type]
    return memory_cls(**config)


def load_memory(file: Union[str, Path]) -> BaseMemory:
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
    return load_memory_from_config(config)


def load_chat_memory_from_config(config: dict) -> ChatMessageHistory:
    """Load ChatMemory from Config Dict."""

    messages = [_message_from_dict(message) for message in config.pop("messages", [])]
    return ChatMessageHistory(messages=messages, **config)
