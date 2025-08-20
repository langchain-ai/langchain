"""
Unified LLM loader for LangChain (OpenAI, OpenRouter, Groq, Ollama).

Example:
    from llm_loader import load_llm

    llm = load_llm("openai", "gpt-4o-mini")
    print(llm.invoke("Hello"))
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Optional imports (avoid breaking if package not installed)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

# Load environment variables from .env
load_dotenv()


def load_llm(
    provider: str,
    model: str,
    temperature: float = 0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Load an LLM wrapper from different providers.

    Args:
        provider: One of ["openai", "openrouter", "groq", "ollama"].
        model: Model name (string).
        temperature: Sampling temperature.
        api_key: Optional API key (overrides .env values).
        base_url: Custom API base URL (used for OpenRouter).

    Returns:
        A LangChain chat model instance.
    """
    provider = provider.lower()

    if provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is not installed.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "openrouter":
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is not installed.")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_base=base_url or "https://openrouter.ai/api/v1",
            openai_api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        )

    elif provider == "groq":
        if ChatGroq is None:
            raise ImportError("langchain_groq is not installed.")
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=api_key or os.getenv("GROQ_API_KEY"),
        )

    elif provider == "ollama":
        if ChatOllama is None:
            raise ImportError("langchain_ollama is not installed.")
        return ChatOllama(
            model=model,
            temperature=temperature,
            validate_model_on_init=True,  # ensures model exists locally
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")
