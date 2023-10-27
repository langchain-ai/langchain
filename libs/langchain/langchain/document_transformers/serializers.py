import inspect
from typing import Any, Callable, Dict

from langchain.schema import BaseDocumentTransformer
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)


def _serialize_transformer(obj: BaseDocumentTransformer) -> Dict:
    return {"_type": type(obj).__name__}


def _serialize_text_splitter(obj: TextSplitter) -> Dict:
    func = obj._length_function
    module = inspect.getmodule(func)
    if module is None:
        raise ImportError(f"Module of function {func.__name__} not found")
    func_dict = {"module": module.__name__, "name": func.__name__}
    return {
        **_serialize_transformer(obj),
        **{
            "chunk_size": obj._chunk_size,
            "chunk_overlap": obj._chunk_overlap,
            "length_function": func_dict,
            "keep_separator": obj._keep_separator,
            "add_start_index": obj._add_start_index,
            "strip_whitespace": obj._strip_whitespace,
        },
    }


def _serialize_char_splitter(obj: CharacterTextSplitter) -> Dict:
    return {
        **_serialize_text_splitter(obj),
        **{
            "separator": obj._separator,
            "is_separator_regex": obj._is_separator_regex,
        },
    }


def _serialize_recursive_splitter(obj: RecursiveCharacterTextSplitter) -> Dict:
    return {
        **_serialize_text_splitter(obj),
        **{
            "separators": obj._separators,
            "keep_separator": obj._keep_separator,
            "is_separator_regex": obj._is_separator_regex,
        },
    }


transformer_to_serializer: Dict[str, Callable[[Any], Dict]] = {
    "CharacterTextSplitter": _serialize_char_splitter,
    "RecursiveCharacterTextSplitter": _serialize_recursive_splitter,
}


def serialize_transformer(transformer: BaseDocumentTransformer) -> Dict:
    transformer_name = type(transformer).__name__
    if transformer_name not in transformer_to_serializer:
        raise ValueError(f"Serializing {transformer_name} transformer not supported")
    return transformer_to_serializer[transformer_name](transformer)
