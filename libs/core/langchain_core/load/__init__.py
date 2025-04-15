"""**Load** module helps with serialization and deserialization."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.load.dump import dumpd, dumps
    from langchain_core.load.load import load, loads
    from langchain_core.load.serializable import Serializable

__all__ = ["dumpd", "dumps", "load", "loads", "Serializable"]

_dynamic_imports = {
    "dumpd": "dump",
    "dumps": "dump",
    "load": "load",
    "loads": "load",
    "Serializable": "serializable",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
