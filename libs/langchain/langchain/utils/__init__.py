"""
**Utility functions** for LangChain.

These functions do not depend on any other LangChain module.
"""

from langchain.utils.dump import default, dumpd, dumps
from langchain.utils.env import get_from_dict_or_env, get_from_env
from langchain.utils.formatting import StrictFormatter, formatter
from langchain.utils.input import (
    get_bolded_text,
    get_color_mapping,
    get_colored_text,
    print_text,
)
from langchain.utils.load import Reviver, loads
from langchain.utils.math import cosine_similarity, cosine_similarity_top_k
from langchain.utils.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedNotImplemented,
    SerializedSecret,
    to_json_not_implemented,
)
from langchain.utils.strings import comma_list, stringify_dict, stringify_value
from langchain.utils.utils import (
    check_package_version,
    get_pydantic_field_names,
    guard_import,
    mock_now,
    raise_for_status_with_text,
    xor_args,
)

__all__ = [
    "Reviver",
    "Serializable",
    "SerializedConstructor",
    "SerializedNotImplemented",
    "SerializedSecret",
    "StrictFormatter",
    "check_package_version",
    "comma_list",
    "cosine_similarity",
    "cosine_similarity_top_k",
    "default",
    "dumpd",
    "dumps",
    "formatter",
    "get_bolded_text",
    "get_color_mapping",
    "get_colored_text",
    "get_from_dict_or_env",
    "get_from_env",
    "get_pydantic_field_names",
    "guard_import",
    "loads",
    "mock_now",
    "print_text",
    "raise_for_status_with_text",
    "stringify_dict",
    "stringify_value",
    "to_json_not_implemented",
    "xor_args",
]
