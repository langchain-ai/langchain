import importlib
import inspect
import pkgutil
from types import ModuleType

from langchain_core.load.mapping import SERIALIZABLE_MAPPING


def import_all_modules(package_name: str) -> dict:
    package = importlib.import_module(package_name)
    classes: dict = {}

    def _handle_module(module: ModuleType) -> None:
        # Iterate over all members of the module

        names = dir(module)

        if hasattr(module, "__all__"):
            names += list(module.__all__)

        names = sorted(set(names))

        for name in names:
            # Check if it's a class or function
            attr = getattr(module, name)

            if not inspect.isclass(attr):
                continue

            if not hasattr(attr, "is_lc_serializable") or not isinstance(attr, type):
                continue

            if (
                isinstance(attr.is_lc_serializable(), bool)  # type: ignore
                and attr.is_lc_serializable()  # type: ignore
            ):
                key = tuple(attr.lc_id())  # type: ignore
                value = tuple(attr.__module__.split(".") + [attr.__name__])
                if key in classes and classes[key] != value:
                    raise ValueError
                classes[key] = value

    _handle_module(package)

    for importer, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            module = importlib.import_module(modname)
        except ModuleNotFoundError:
            continue
        _handle_module(module)

    return classes


def test_import_all_modules() -> None:
    """Test import all modules works as expected"""
    all_modules = import_all_modules("langchain")
    filtered_modules = [
        k
        for k in all_modules
        if len(k) == 4 and tuple(k[:2]) == ("langchain", "chat_models")
    ]
    # This test will need to be updated if new serializable classes are added
    # to community
    assert sorted(filtered_modules) == sorted(
        [
            ("langchain", "chat_models", "azure_openai", "AzureChatOpenAI"),
            ("langchain", "chat_models", "bedrock", "BedrockChat"),
            ("langchain", "chat_models", "anthropic", "ChatAnthropic"),
            ("langchain", "chat_models", "fireworks", "ChatFireworks"),
            ("langchain", "chat_models", "google_palm", "ChatGooglePalm"),
            ("langchain", "chat_models", "openai", "ChatOpenAI"),
            ("langchain", "chat_models", "vertexai", "ChatVertexAI"),
            ("langchain", "chat_models", "mistralai", "ChatMistralAI"),
        ]
    )


def test_serializable_mapping() -> None:
    to_skip = {
        # This should have had a different namespace, as it was never
        # exported from the langchain module, but we keep for whoever has
        # already serialized it.
        ("langchain", "prompts", "image", "ImagePromptTemplate"): (
            "langchain_core",
            "prompts",
            "image",
            "ImagePromptTemplate",
        ),
        # This is not exported from langchain, only langchain_core
        ("langchain_core", "prompts", "structured", "StructuredPrompt"): (
            "langchain_core",
            "prompts",
            "structured",
            "StructuredPrompt",
        ),
        # This is not exported from langchain, only langchain_core
        ("langchain", "schema", "messages", "RemoveMessage"): (
            "langchain_core",
            "messages",
            "modifier",
            "RemoveMessage",
        ),
        ("langchain", "chat_models", "mistralai", "MistralAI"): (
            "langchain_mistralai",
            "chat_models",
            "ChatMistralAI",
        ),
        ("langchain_groq", "chat_models", "ChatGroq"): (
            "langchain_groq",
            "chat_models",
            "ChatGroq",
        ),
    }
    serializable_modules = import_all_modules("langchain")

    missing = set(SERIALIZABLE_MAPPING).difference(
        set(serializable_modules).union(to_skip)
    )
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
