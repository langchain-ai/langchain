# ruff: noqa: E402
"""Main entrypoint into package."""
import warnings
from importlib import metadata
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain.schema import BaseCache


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

verbose: bool = False
debug: bool = False
llm_cache: Optional["BaseCache"] = None


def __getattr__(name):
    warnings.warn(f"Importing {name} is no longer supported.")
    if name in ["MRKLChain", "ReActChain", "SelfAskWithSearchChain"]:
        from langchain.agents import MRKLChain, ReActChain, SelfAskWithSearchChain

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in [
        "ConversationChain",
        "LLMBashChain",
        "LLMChain",
        "LLMCheckerChain",
        "LLMMathChain",
        "QAWithSourcesChain",
        "VectorDBQA",
        "VectorDBQAWithSourcesChain",
    ]:
        from langchain.chains import (
            ConversationChain,
            LLMBashChain,
            LLMChain,
            LLMCheckerChain,
            LLMMathChain,
            QAWithSourcesChain,
            VectorDBQA,
            VectorDBQAWithSourcesChain,
        )

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in ["InMemoryDocstore", "Wikipedia"]:
        from langchain.docstore import InMemoryDocstore, Wikipedia

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in [
        "Anthropic",
        "Banana",
        "CerebriumAI",
        "Cohere",
        "ForefrontAI",
        "GooseAI",
        "HuggingFaceHub",
        "HuggingFaceTextGenInference",
        "LlamaCpp",
        "Modal",
        "OpenAI",
        "Petals",
        "PipelineAI",
        "SagemakerEndpoint",
        "StochasticAI",
        "Writer",
    ]:
        from langchain.llms import (
            Anthropic,
            Banana,
            CerebriumAI,
            Cohere,
            ForefrontAI,
            GooseAI,
            HuggingFaceHub,
            HuggingFaceTextGenInference,
            LlamaCpp,
            Modal,
            OpenAI,
            Petals,
            PipelineAI,
            SagemakerEndpoint,
            StochasticAI,
            Writer,
        )

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in ["HuggingFacePipeline"]:
        from langchain.llms.huggingface_pipeline import HuggingFacePipeline

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in [
        "FewShotPromptTemplate",
        "Prompt",
        "PromptTemplate",
    ]:
        from langchain.prompts import (
            FewShotPromptTemplate,
            Prompt,
            PromptTemplate,
        )

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in ["BasePromptTemplate"]:
        from langchain.schema.prompt_template import BasePromptTemplate

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in [
        "ArxivAPIWrapper",
        "GoldenQueryAPIWrapper",
        "GoogleSearchAPIWrapper",
        "GoogleSerperAPIWrapper",
        "PowerBIDataset",
        "SearxSearchWrapper",
        "SerpAPIWrapper",
        "WikipediaAPIWrapper",
        "WolframAlphaAPIWrapper",
        "SQLDatabase",
    ]:
        from langchain.utilities import (
            ArxivAPIWrapper,
            GoldenQueryAPIWrapper,
            GoogleSearchAPIWrapper,
            GoogleSerperAPIWrapper,
            PowerBIDataset,
            SearxSearchWrapper,
            SerpAPIWrapper,
            SQLDatabase,
            WikipediaAPIWrapper,
            WolframAlphaAPIWrapper,
        )

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    elif name in ["FAISS", "ElasticVectorSearch"]:
        from langchain.vectorstores import FAISS, ElasticVectorSearch

        # Add the imported attribute to the module's namespace
        # This avoids re-entering this in the future
        globals()[name] = locals()[name]
        return locals()[name]
    # For backwards compatibility
    elif name == "SerpAPIChain":
        from langchain.utilities import SerpAPIWrapper

        return SerpAPIWrapper
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = [
    "LLMChain",
    "LLMBashChain",
    "LLMCheckerChain",
    "LLMMathChain",
    "ArxivAPIWrapper",
    "GoldenQueryAPIWrapper",
    "SelfAskWithSearchChain",
    "SerpAPIWrapper",
    "SerpAPIChain",
    "SearxSearchWrapper",
    "GoogleSearchAPIWrapper",
    "GoogleSerperAPIWrapper",
    "WolframAlphaAPIWrapper",
    "WikipediaAPIWrapper",
    "Anthropic",
    "Banana",
    "CerebriumAI",
    "Cohere",
    "ForefrontAI",
    "GooseAI",
    "Modal",
    "OpenAI",
    "Petals",
    "PipelineAI",
    "StochasticAI",
    "Writer",
    "BasePromptTemplate",
    "Prompt",
    "FewShotPromptTemplate",
    "PromptTemplate",
    "ReActChain",
    "Wikipedia",
    "HuggingFaceHub",
    "SagemakerEndpoint",
    "HuggingFacePipeline",
    "SQLDatabase",
    "PowerBIDataset",
    "FAISS",
    "MRKLChain",
    "VectorDBQA",
    "ElasticVectorSearch",
    "InMemoryDocstore",
    "ConversationChain",
    "VectorDBQAWithSourcesChain",
    "QAWithSourcesChain",
    "LlamaCpp",
    "HuggingFaceTextGenInference",
]
