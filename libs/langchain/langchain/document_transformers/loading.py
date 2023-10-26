from typing import Type, TypeVar

from langchain.schema import BaseDocumentTransformer
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

T = TypeVar("T", bound=TextSplitter)


def _load_text_splitter(splitter: Type[T], config: dict) -> T:
    length_function = len
    if "length_function" in config:
        func_config = config.pop("length_function")
        if "module" not in func_config or "name" not in func_config:
            raise ValueError(
                "`module` or `name` is required properties in length_function config"
            )
        try:
            module = __import__(func_config["module"])
            length_function = getattr(module, func_config["name"])
        except ModuleNotFoundError:
            raise ValueError(f'Module {func_config["module"]} not found')
        except AttributeError:
            raise ValueError(
                f'Function {func_config["name"]} not found '
                f'in module {func_config["module"]}'
            )
    return splitter(length_function=length_function, **config)


def _load_char_splitter(config: dict) -> CharacterTextSplitter:
    return _load_text_splitter(CharacterTextSplitter, config)


def _load_recursive_splitter(config: dict) -> RecursiveCharacterTextSplitter:
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
