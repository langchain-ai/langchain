import importlib
import pkgutil

from langchain_core.load.mapping import SERIALIZABLE_MAPPING


def import_all_modules(package_name: str) -> dict:
    package = importlib.import_module(package_name)
    classes: dict = {}

    for attribute_name in dir(package):
        attribute = getattr(package, attribute_name)
        if hasattr(attribute, "is_lc_serializable") and isinstance(attribute, type):
            if (
                isinstance(attribute.is_lc_serializable(), bool)  # type: ignore
                and attribute.is_lc_serializable()  # type: ignore
            ):
                key = tuple(attribute.lc_id())  # type: ignore
                value = tuple(attribute.__module__.split(".") + [attribute.__name__])
                if key in classes and classes[key] != value:
                    raise ValueError
                classes[key] = value
    if hasattr(package, "__path__"):
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            package.__path__, package_name + "."
        ):
            if module_name not in (
                "langchain.chains.llm_bash",
                "langchain.chains.llm_symbolic_math",
                "langchain_community.tools.python",
                "langchain_community.vectorstores._pgvector_data_models",
            ):
                importlib.import_module(module_name)
                new_classes = import_all_modules(module_name)
                for k, v in new_classes.items():
                    if k in classes and classes[k] != v:
                        raise ValueError
                    classes[k] = v
    return classes


def test_serializable_mapping() -> None:
    serializable_modules = import_all_modules("langchain")
    missing = set(SERIALIZABLE_MAPPING).difference(serializable_modules)
    assert missing == set()
    extra = set(serializable_modules).difference(SERIALIZABLE_MAPPING)
    assert extra == set()

    for k, import_path in serializable_modules.items():
        import_dir, import_obj = import_path[:-1], import_path[-1]
        # Import module
        mod = importlib.import_module(".".join(import_dir))
        # Import class
        cls = getattr(mod, import_obj)
        assert list(k) == cls.lc_id()
