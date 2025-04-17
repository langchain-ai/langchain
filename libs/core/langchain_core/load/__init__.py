"""**Load** module helps with serialization and deserialization."""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.load.dump import dumpd, dumps
    from langchain_core.load.load import loads
    from langchain_core.load.serializable import Serializable

# Unfortunately, we have to eagerly import load from langchain_core/load/load.py
# eagerly to avoid a namespace conflict. We want users to still be able to use
# `from langchain_core.load import load` to get the load function, but
# the `from langchain_core.load.load import load` absolute import should also work.
from langchain_core.load.load import load

__all__ = ("dumpd", "dumps", "load", "loads", "Serializable")

_dynamic_imports = {
    "dumpd": "dump",
    "dumps": "dump",
    "loads": "load",
    "Serializable": "serializable",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
