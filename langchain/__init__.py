"""Main entrypoint into package."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()

from langchain.chains import (
    LLMChain,
    LLMMathChain,
    PythonChain,
    ReActChain,
    SelfAskWithSearchChain,
    SerpAPIChain,
    SelfConsistencyChain,
)
from langchain.docstore import Wikipedia
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.prompt import Prompt

__all__ = [
    "LLMChain",
    "LLMMathChain",
    "PythonChain",
    "SelfAskWithSearchChain",
    "SelfConsistencyChain",
    "SerpAPIChain",
    "Cohere",
    "OpenAI",
    "Prompt",
    "ReActChain",
    "Wikipedia",
    "HuggingFaceHub",
]
