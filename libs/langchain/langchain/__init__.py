# ruff: noqa: E402
"""Main entrypoint into package."""
import warnings
from importlib import metadata
from typing import Any, Optional

from langchain_core._api.deprecation import surface_langchain_deprecation_warnings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


def _warn_on_import(name: str, replacement: Optional[str] = None) -> None:
    """Warn on import of deprecated module."""
    from langchain.utils.interactive_env import is_interactive_env

    if is_interactive_env():
        # No warnings for interactive environments.
        # This is done to avoid polluting the output of interactive environments
        # where users rely on auto-complete and may trigger this warning
        # even if they are not using any deprecated modules
        return

    if replacement:
        warnings.warn(
            f"Importing {name} from langchain root module is no longer supported. "
            f"Please use {replacement} instead."
        )
    else:
        warnings.warn(
            f"Importing {name} from langchain root module is no longer supported."
        )


# Surfaces Deprecation and Pending Deprecation warnings from langchain.
surface_langchain_deprecation_warnings()


def _raise_community_deprecation_error(name: str, new_module: str) -> None:
    raise ImportError(
        f"{name} has been moved to the langchain-community package. "
        f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
        f"information.\n\nTo use it install langchain-community:\n\n"
        f"`pip install -U langchain-community`\n\n"
        f"then import with:\n\n"
        f"`from {new_module} import {name}`"
    )


def __getattr__(name: str) -> Any:
    if name == "MRKLChain":
        from langchain.agents import MRKLChain

        _warn_on_import(name, replacement="langchain.agents.MRKLChain")

        return MRKLChain
    elif name == "ReActChain":
        from langchain.agents import ReActChain

        _warn_on_import(name, replacement="langchain.agents.ReActChain")

        return ReActChain
    elif name == "SelfAskWithSearchChain":
        from langchain.agents import SelfAskWithSearchChain

        _warn_on_import(name, replacement="langchain.agents.SelfAskWithSearchChain")

        return SelfAskWithSearchChain
    elif name == "ConversationChain":
        from langchain.chains import ConversationChain

        _warn_on_import(name, replacement="langchain.chains.ConversationChain")

        return ConversationChain
    elif name == "LLMBashChain":
        raise ImportError(
            "This module has been moved to langchain-experimental. "
            "For more details: "
            "https://github.com/langchain-ai/langchain/discussions/11352."
            "To access this code, install it with `pip install langchain-experimental`."
            "`from langchain_experimental.llm_bash.base "
            "import LLMBashChain`"
        )

    elif name == "LLMChain":
        from langchain.chains import LLMChain

        _warn_on_import(name, replacement="langchain.chains.LLMChain")

        return LLMChain
    elif name == "LLMCheckerChain":
        from langchain.chains import LLMCheckerChain

        _warn_on_import(name, replacement="langchain.chains.LLMCheckerChain")

        return LLMCheckerChain
    elif name == "LLMMathChain":
        from langchain.chains import LLMMathChain

        _warn_on_import(name, replacement="langchain.chains.LLMMathChain")

        return LLMMathChain
    elif name == "QAWithSourcesChain":
        from langchain.chains import QAWithSourcesChain

        _warn_on_import(name, replacement="langchain.chains.QAWithSourcesChain")

        return QAWithSourcesChain
    elif name == "VectorDBQA":
        from langchain.chains import VectorDBQA

        _warn_on_import(name, replacement="langchain.chains.VectorDBQA")

        return VectorDBQA
    elif name == "VectorDBQAWithSourcesChain":
        from langchain.chains import VectorDBQAWithSourcesChain

        _warn_on_import(name, replacement="langchain.chains.VectorDBQAWithSourcesChain")

        return VectorDBQAWithSourcesChain
    elif name == "InMemoryDocstore":
        from langchain.docstore import InMemoryDocstore

        _warn_on_import(name, replacement="langchain.docstore.InMemoryDocstore")

        return InMemoryDocstore
    elif name == "Wikipedia":
        from langchain.docstore import Wikipedia

        _warn_on_import(name, replacement="langchain.docstore.Wikipedia")

        return Wikipedia
    elif name == "Anthropic":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "Banana":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "CerebriumAI":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "Cohere":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "ForefrontAI":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "GooseAI":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "HuggingFaceHub":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "HuggingFaceTextGenInference":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "LlamaCpp":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "Modal":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "OpenAI":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "Petals":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "PipelineAI":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "SagemakerEndpoint":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "StochasticAI":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "Writer":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "HuggingFacePipeline":
        _raise_community_deprecation_error(name, "langchain_community.llms")
    elif name == "FewShotPromptTemplate":
        from langchain_core.prompts import FewShotPromptTemplate

        _warn_on_import(
            name, replacement="langchain_core.prompts.FewShotPromptTemplate"
        )

        return FewShotPromptTemplate
    elif name == "Prompt":
        from langchain_core.prompts import PromptTemplate

        _warn_on_import(name, replacement="langchain_core.prompts.PromptTemplate")

        # it's renamed as prompt template anyways
        # this is just for backwards compat
        return PromptTemplate
    elif name == "PromptTemplate":
        from langchain_core.prompts import PromptTemplate

        _warn_on_import(name, replacement="langchain_core.prompts.PromptTemplate")

        return PromptTemplate
    elif name == "BasePromptTemplate":
        from langchain_core.prompts import BasePromptTemplate

        _warn_on_import(name, replacement="langchain_core.prompts.BasePromptTemplate")

        return BasePromptTemplate
    elif name == "ArxivAPIWrapper":
        from langchain_community.utilities import ArxivAPIWrapper

        _warn_on_import(
            name, replacement="langchain_community.utilities.ArxivAPIWrapper"
        )

        return ArxivAPIWrapper
    elif name == "GoldenQueryAPIWrapper":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "GoogleSearchAPIWrapper":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "GoogleSerperAPIWrapper":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "PowerBIDataset":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "SearxSearchWrapper":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "WikipediaAPIWrapper":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "WolframAlphaAPIWrapper":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "SQLDatabase":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "FAISS":
        _raise_community_deprecation_error(name, "langchain_community.vectorstores")
    elif name == "ElasticVectorSearch":
        _raise_community_deprecation_error(name, "langchain_community.vectorstores")
    # For backwards compatibility
    elif name == "SerpAPIChain" or name == "SerpAPIWrapper":
        _raise_community_deprecation_error(name, "langchain_community.utilities")
    elif name == "verbose":
        from langchain.globals import _verbose

        _warn_on_import(
            name,
            replacement=(
                "langchain.globals.set_verbose() / langchain.globals.get_verbose()"
            ),
        )

        return _verbose
    elif name == "debug":
        from langchain.globals import _debug

        _warn_on_import(
            name,
            replacement=(
                "langchain.globals.set_debug() / langchain.globals.get_debug()"
            ),
        )

        return _debug
    elif name == "llm_cache":
        from langchain.globals import _llm_cache

        _warn_on_import(
            name,
            replacement=(
                "langchain.globals.set_llm_cache() / langchain.globals.get_llm_cache()"
            ),
        )

        return _llm_cache
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = [
    "LLMChain",
    "LLMCheckerChain",
    "LLMMathChain",
    "SelfAskWithSearchChain",
    "BasePromptTemplate",
    "Prompt",
    "FewShotPromptTemplate",
    "PromptTemplate",
    "ReActChain",
    "MRKLChain",
    "VectorDBQA",
    "ConversationChain",
    "VectorDBQAWithSourcesChain",
    "QAWithSourcesChain",
]
