"""
**Utility functions** for LangChain.

These functions do not depend on any other LangChain module.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "image":
        import langchain_core.utils.image

        return langchain_core.utils.image
    elif name == "get_from_dict_or_env":
        from langchain_core.utils.env import get_from_dict_or_env

        return get_from_dict_or_env
    elif name == "get_from_env":
        from langchain_core.utils.env import get_from_env

        return get_from_env
    elif name == "StrictFormatter":
        from langchain_core.utils.formatting import StrictFormatter

        return StrictFormatter
    elif name == "formatter":
        from langchain_core.utils.formatting import formatter

        return formatter
    elif name == "get_bolded_text":
        from langchain_core.utils.input import get_bolded_text

        return get_bolded_text
    elif name == "get_color_mapping":
        from langchain_core.utils.input import get_color_mapping

        return get_color_mapping
    elif name == "get_colored_text":
        from langchain_core.utils.input import get_colored_text

        return get_colored_text
    elif name == "print_text":
        from langchain_core.utils.input import print_text

        return print_text
    elif name == "try_load_from_hub":
        from langchain_core.utils.loading import try_load_from_hub

        return try_load_from_hub
    elif name == "comma_list":
        from langchain_core.utils.strings import comma_list

        return comma_list
    elif name == "stringify_dict":
        from langchain_core.utils.strings import stringify_dict

        return stringify_dict
    elif name == "stringify_value":
        from langchain_core.utils.strings import stringify_value

        return stringify_value
    elif name == "build_extra_kwargs":
        from langchain_core.utils.utils import build_extra_kwargs

        return build_extra_kwargs
    elif name == "check_package_version":
        from langchain_core.utils.utils import check_package_version

        return check_package_version
    elif name == "convert_to_secret_str":
        from langchain_core.utils.utils import convert_to_secret_str

        return convert_to_secret_str
    elif name == "get_pydantic_field_names":
        from langchain_core.utils.utils import get_pydantic_field_names

        return get_pydantic_field_names
    elif name == "guard_import":
        from langchain_core.utils.utils import guard_import

        return guard_import
    elif name == "mock_now":
        from langchain_core.utils.utils import mock_now

        return mock_now
    elif name == "raise_for_status_with_text":
        from langchain_core.utils.utils import raise_for_status_with_text

        return raise_for_status_with_text
    elif name == "xor_args":
        from langchain_core.utils.utils import xor_args

        return xor_args
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "StrictFormatter",
    "check_package_version",
    "convert_to_secret_str",
    "formatter",
    "get_bolded_text",
    "get_color_mapping",
    "get_colored_text",
    "get_pydantic_field_names",
    "guard_import",
    "mock_now",
    "print_text",
    "raise_for_status_with_text",
    "xor_args",
    "try_load_from_hub",
    "build_extra_kwargs",
    "image",
    "get_from_env",
    "get_from_dict_or_env",
    "stringify_dict",
    "comma_list",
    "stringify_value",
]
