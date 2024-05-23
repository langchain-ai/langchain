import importlib
from pathlib import Path

# Attempt to recursively import all modules in langchain
PKG_ROOT = Path(__file__).parent.parent.parent

COMMUNITY_NOT_INSTALLED = importlib.util.find_spec("langchain_community") is None


def test_import_all() -> None:
    """Generate the public API for this package."""
    library_code = PKG_ROOT / "langchain"
    for path in library_code.rglob("*.py"):
        # Calculate the relative path to the module
        module_name = (
            path.relative_to(PKG_ROOT).with_suffix("").as_posix().replace("/", ".")
        )
        if module_name.endswith("__init__"):
            # Without init
            module_name = module_name.rsplit(".", 1)[0]

        mod = importlib.import_module(module_name)

        all = getattr(mod, "__all__", [])

        for name in all:
            # Attempt to import the name from the module
            try:
                obj = getattr(mod, name)
                assert obj is not None
            except ModuleNotFoundError as e:
                # If the module is not installed, we suppress the error
                if "Module langchain_community" in str(e) and COMMUNITY_NOT_INSTALLED:
                    pass


def test_import_all_using_dir() -> None:
    """Generate the public API for this package."""
    library_code = PKG_ROOT / "langchain"
    for path in library_code.rglob("*.py"):
        # Calculate the relative path to the module
        module_name = (
            path.relative_to(PKG_ROOT).with_suffix("").as_posix().replace("/", ".")
        )
        if module_name.endswith("__init__"):
            # Without init
            module_name = module_name.rsplit(".", 1)[0]

        if module_name.startswith("langchain_community.") and COMMUNITY_NOT_INSTALLED:
            continue

        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Could not import {module_name}") from e
        all = dir(mod)

        for name in all:
            if name.strip().startswith("_"):
                continue
            # Attempt to import the name from the module
            getattr(mod, name)
