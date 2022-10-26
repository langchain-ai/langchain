"""Main entrypoint into package."""

from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()

from langchain.chains import (
    LLMChain,
    LLMMathChain,
    PythonChain,
    SelfAskWithSearchChain,
    SerpAPIChain,
)
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.prompt import Prompt

__all__ = [
    "LLMChain",
    "LLMMathChain",
    "PythonChain",
    "SelfAskWithSearchChain",
    "SerpAPIChain",
    "Cohere",
    "OpenAI",
    "Prompt",
    "HuggingFaceHub",
]
