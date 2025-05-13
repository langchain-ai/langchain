"""**Utility functions** for LangChain.

These functions do not depend on any other LangChain module.
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    # for type checking and IDE support, we include the imports here
    # but we don't want to eagerly import them at runtime
    from langchain_core.utils import image
    from langchain_core.utils.aiter import abatch_iterate
    from langchain_core.utils.env import get_from_dict_or_env, get_from_env
    from langchain_core.utils.formatting import StrictFormatter, formatter
    from langchain_core.utils.input import (
        get_bolded_text,
        get_color_mapping,
        get_colored_text,
        print_text,
    )
    from langchain_core.utils.iter import batch_iterate
    from langchain_core.utils.loading import try_load_from_hub
    from langchain_core.utils.pydantic import pre_init
    from langchain_core.utils.strings import comma_list, stringify_dict, stringify_value
    from langchain_core.utils.utils import (
        build_extra_kwargs,
        check_package_version,
        convert_to_secret_str,
        from_env,
        get_pydantic_field_names,
        guard_import,
        mock_now,
        raise_for_status_with_text,
        secret_from_env,
        xor_args,
    )

__all__ = (
    "build_extra_kwargs",
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
    "image",
    "get_from_env",
    "get_from_dict_or_env",
    "stringify_dict",
    "comma_list",
    "stringify_value",
    "pre_init",
    "batch_iterate",
    "abatch_iterate",
    "from_env",
    "secret_from_env",
)

_dynamic_imports = {
    "image": "__module__",
    "abatch_iterate": "aiter",
    "get_from_dict_or_env": "env",
    "get_from_env": "env",
    "StrictFormatter": "formatting",
    "formatter": "formatting",
    "get_bolded_text": "input",
    "get_color_mapping": "input",
    "get_colored_text": "input",
    "print_text": "input",
    "batch_iterate": "iter",
    "try_load_from_hub": "loading",
    "pre_init": "pydantic",
    "comma_list": "strings",
    "stringify_dict": "strings",
    "stringify_value": "strings",
    "build_extra_kwargs": "utils",
    "check_package_version": "utils",
    "convert_to_secret_str": "utils",
    "from_env": "utils",
    "get_pydantic_field_names": "utils",
    "guard_import": "utils",
    "mock_now": "utils",
    "secret_from_env": "utils",
    "xor_args": "utils",
    "raise_for_status_with_text": "utils",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
