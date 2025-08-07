def __getattr__(_: str = "") -> None:
    """Raise an error on import since is deprecated."""
    msg = (
        "This module has been moved to langchain-experimental. "
        "For more details: https://github.com/langchain-ai/langchain/discussions/11352."
        "To access this code, install it with `pip install langchain-experimental`."
        "`from langchain_experimental.llm_bash.base "
        "import LLMBashChain`"
    )
    raise AttributeError(msg)
