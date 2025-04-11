"""**Load** module helps with serialization and deserialization."""

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

if TYPE_CHECKING:
    from langchain_core.load.dump import dumpd, dumps
    from langchain_core.load.load import load, loads
    from langchain_core.load.serializable import Serializable

__all__ = ["dumpd", "dumps", "load", "loads", "Serializable"]

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="load",
    dynamic_imports={
        "dumpd": "dump",
        "dumps": "dump",
        "load": "load",
        "loads": "load",
        "Serializable": "serializable",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
