from langchain._internal.lazy_import import import_attr

# Dynamic imports mapping
_dynamic_imports = {
    "create_iterative_extractor": "extraction",
    "create_map_reduce_extractor": "extraction",
    "create_recursive_extractor": "extraction",
}

__all__ = [
    "create_iterative_extractor",
    "create_map_reduce_extractor",
    "create_recursive_extractor",
]


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
