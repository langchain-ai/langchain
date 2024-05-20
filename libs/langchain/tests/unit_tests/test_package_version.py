import importlib

from packaging.version import Version, parse


def test_package_version_is_parsable() -> None:
    module = importlib.import_module("langchain")
    assert isinstance(parse(module.__version__), Version)
