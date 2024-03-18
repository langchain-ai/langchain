from typing import Any

DEPRECATED_IMPORTS = [
    "import_spacy",
    "import_pandas",
    "import_textstat",
    "_flatten_dict",
    "flatten_dict",
    "hash_string",
    "load_json",
    "BaseMetadataCallbackHandler",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.callbacks.utils import {name}`"
        )

    raise AttributeError()
