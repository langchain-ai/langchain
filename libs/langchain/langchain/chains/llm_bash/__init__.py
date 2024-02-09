def __getattr__(name: str = "") -> None:
    """Raise an error on import since is deprecated."""
    raise ImportError(
        "This module has been moved to gigachain-experimental. "
        "For more details: https://github.com/langchain-ai/langchain/discussions/11352."
        "To access this code, install it with `pip install gigachain-experimental`."
        "`from langchain_experimental.llm_bash.base "
        "import LLMBashChain`"
    )
