"""Internal utilities for lazy loading of modules in __init__.py files."""

from importlib import import_module
from typing import Callable


def create_dynamic_getattr(
    package_name: str, module_path: str, dynamic_imports: dict[str, str]
) -> Callable[[str], object]:
    """Create a dynamic getattr function for lazy loading of module attributes.

    Args:
        package_name: The name of the package, e.g., "langchain_core".
        module_path: The path to the module, e.g., "utils".
        dynamic_imports: Mapping of attr names to their module names.
    """

    def _dynamic_getattr(attr_name: str) -> object:
        module_name = dynamic_imports.get(attr_name)
        if module_name is None:
            import_error_msg = (
                f"cannot import name '{attr_name}' from '{package_name}.{module_path}'"
            )
            raise ImportError(import_error_msg)

        if module_name == "__module__":
            attr = import_module(f".{module_path}.{attr_name}", package=package_name)
            globals()[attr_name] = attr
            return attr

        module = import_module(f".{module_path}.{module_name}", package=package_name)
        attr = getattr(module, attr_name)

        g = globals()
        g[attr_name] = attr

        # TODO: determine if we want to do this eagerly
        # for k, v_module_name in dynamic_imports.items():
        #     if v_module_name == module_name:
        #         g[k] = getattr(module, k)

        return attr

    return _dynamic_getattr
