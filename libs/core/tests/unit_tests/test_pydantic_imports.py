import importlib
from pathlib import Path

from pydantic import BaseModel


def test_all_models_built() -> None:
    for path in Path("../core/langchain_core/").glob("*"):
        module_name = path.stem
        if not module_name.startswith(".") and path.suffix != ".typed":
            module = importlib.import_module("langchain_core." + module_name)
            all_ = getattr(module, "__all__", [])
            for attr_name in all_:
                attr = getattr(module, attr_name)
                try:
                    if issubclass(attr, BaseModel):
                        assert attr.__pydantic_complete__ is True
                except TypeError:
                    # This is expected for non-class attributes
                    pass
