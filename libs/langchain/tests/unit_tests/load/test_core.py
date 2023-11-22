import importlib

from langchain_core.load import Serializable

core_modules = [
    "agents",
    "caches",
    "callbacks",
    "chat_history",
    "chat_sessions",
    "document_transformers",
    "documents",
    "embeddings",
    "env",
    "example_selectors",
    "exceptions",
    "globals",
    "language_models",
    "load",
    "memory",
    "messages",
    "output_parsers",
    "outputs",
    "prompt_values",
    "prompts",
    "pydantic_v1",
    "retrievers",
    "runnables",
    "stores",
    "tools",
    "tracers",
    "utils",
    "vectorstores",
]


def test_core_exported_from_langchain() -> None:
    # iterate through core modules and get exported names that inherit from serializable
    # and are not private
    wrong_module = []
    does_not_exist = []

    for module_name in core_modules:
        module = importlib.import_module(f"langchain_core.{module_name}")
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if not isinstance(obj, type):
                continue
            if not issubclass(obj, Serializable):
                continue
            obj_name = f"langchain_core.{module_name}.{name}"
            lc_id = obj.lc_id()  # type: ignore
            if not lc_id[0] == "langchain":
                wrong_module.append(f"{obj_name} -> {lc_id}")
                continue
            # see if importable
            [*id_namespace, id_name] = lc_id
            import_name = ".".join(id_namespace)
            try:
                import_module = importlib.import_module(import_name)
                import_obj = getattr(import_module, id_name)
            except (ImportError, AttributeError):
                does_not_exist.append(f"{obj_name} -> {lc_id}")
                continue

            # assert same id
            assert import_obj.lc_id() == lc_id, f"{obj_name} -> {lc_id}"
            # assert serializable
            assert issubclass(
                import_obj, Serializable
            ), f"Referenced object not serializable: {obj_name} -> {lc_id}"
    if len(wrong_module) == 0 and len(does_not_exist) == 0:
        return

    wrong_module_message = "\n".join(f"- {m}" for m in wrong_module) or "None! Passed"
    does_not_exist_message = (
        "\n".join(f"- {m}" for m in does_not_exist) or "None! Passed"
    )
    assert False, f"""LC ID must be from langchain.x ({len(wrong_module)}):
{wrong_module_message}

The following LC IDs do not exist ({len(does_not_exist)}):
{does_not_exist_message}"""
