from typing import Type

from langchain.schema import BaseDocumentTransformer
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)


def _load_text_splitter(splitter: Type[TextSplitter], config: dict) -> TextSplitter:
    return splitter(**config)


def _load_char_splitter(config: dict) -> TextSplitter:
    return _load_text_splitter(CharacterTextSplitter, config)


def _load_recursive_splitter(config: dict) -> TextSplitter:
    return _load_text_splitter(RecursiveCharacterTextSplitter, config)


transformer_to_loader = {
    "CharacterTextSplitter": _load_char_splitter,
    "RecursiveCharacterTextSplitter": _load_recursive_splitter,
}


def load_transformer_from_config(config: dict) -> BaseDocumentTransformer:
    """Load chain from Config Dict."""
    if "_type" not in config:
        raise ValueError("Must specify a document transformer type in config")
    config_type = config.pop("_type")

    if config_type not in transformer_to_loader:
        raise ValueError(f"Loading {config_type} document transformer not supported")

    transformer_loader = transformer_to_loader[config_type]
    return transformer_loader(config)
