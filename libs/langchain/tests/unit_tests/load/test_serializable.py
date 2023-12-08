import pkgutil
import importlib
from langchain_core.load.mapping import SERIALIZABLE_MAPPING


def import_all_modules(package_name):
    package = importlib.import_module(package_name)
    classes = []

    for attribute_name in dir(package):
        attribute = getattr(package, attribute_name)
        if hasattr(attribute, "is_lc_serializable"):
            if isinstance(attribute.is_lc_serializable(), bool) and attribute.is_lc_serializable():
                classes.append(tuple(attribute.lc_id()))
    if hasattr(package, "__path__"):
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package_name + '.'):
            if module_name not in (
                    "langchain.chains.llm_bash",
                    "langchain.chains.llm_symbolic_math",
                    "langchain.tools.python",
                    "langchain.vectorstores._pgvector_data_models",
            ):
                importlib.import_module(module_name)
                classes.extend(import_all_modules(module_name))
    return list(set(classes))


def test_serializable_mapping():
    serializable_modules = set(import_all_modules("langchain"))
    extra_modules = serializable_modules.difference(SERIALIZABLE_MAPPING)
    assert len(extra_modules) == 0
    missing_modules = set(SERIALIZABLE_MAPPING).difference(serializable_modules)
    assert len(missing_modules) == 0
