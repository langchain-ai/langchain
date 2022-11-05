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
    SQLDatabaseChain,
)
from langchain.docstore import Wikipedia
from langchain.example_generator import generate_example  # noqa
from langchain.example_generator import generate_example_from_dynamic_prompt  # noqa
from langchain.faiss import FAISS
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.prompts import BasePrompt, DynamicPrompt, Prompt
from langchain.sql_database import SQLDatabase

__all__ = [
    "LLMChain",
    "LLMMathChain",
    "PythonChain",
    "SelfAskWithSearchChain",
    "SerpAPIChain",
    "Cohere",
    "OpenAI",
    "BasePrompt",
    "DynamicPrompt",
    "Prompt",
    "ReActChain",
    "Wikipedia",
    "HuggingFaceHub",
    "SQLDatabase",
    "SQLDatabaseChain",
    "FAISS",
]
