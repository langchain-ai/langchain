"""
Functions that support shorthand access of LangChain classes like

::code
    import langchain_core as lc

    model = lc.chat_model("claude-3-5-sonnet-20240620", provider="anthropic")
"""

import importlib
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.tools import BaseTool, BaseToolkit
    from langchain_core.vectorstores import VectorStore

ProviderManifestType = dict[
    Literal[
        "chat_model",
        "retriever",
        "embeddings",
        "vectorstore",
        "tool",
        "toolkit",
        "document_loader",
    ],
    list[str],
]
_registered_providers: dict[
    str,
    ProviderManifestType,
] = {}


def register_manifest(
    provider: str,
    provider_manifest: ProviderManifestType,
    *,
    dangerously_allow_provider_overwrite: bool = False,
) -> None:
    if provider in _registered_providers and not dangerously_allow_provider_overwrite:
        msg = (
            f"Provider `{provider}` was already registered. To allow overwriting, "
            "pass `dangerously_allow_provider_overwrite=True`"
        )
        raise ValueError(msg)
    _registered_providers[provider] = provider_manifest


def register(package_root: str) -> None:
    manifest_module = importlib.import_module("lc_manifest", package_root)
    if not hasattr(manifest_module, "manifest"):
        msg = (
            f"Package registration requires a {package_root}.lc_manifest.manifest "
            "dictionary to be declared"
        )
        raise ValueError(msg)

    manifest = cast(dict[str, ProviderManifestType], manifest_module.manifest)
    for provider, provider_manifest in manifest.items():
        register_manifest(provider, provider_manifest)


def chat_model(model: str, *, provider: str | None = None, **kwargs) -> BaseChatModel:
    pass


def retriever(name: str, **kwargs) -> BaseRetriever:
    pass


def tool(name: str, **kwargs) -> BaseTool:
    pass


def toolkit(name: str, **kwargs) -> BaseToolkit:
    pass


def document_loader(name: str, **kwargs) -> BaseLoader:
    pass


def vectorstore(name: str, **kwargs) -> VectorStore:
    pass


def embeddings(model: str, *, provider: str | None = None, **kwargs) -> Embeddings:
    pass
