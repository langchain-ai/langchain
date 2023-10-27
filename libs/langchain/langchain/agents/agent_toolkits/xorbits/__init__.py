def raise_on_import() -> None:
    """Raise on import letting users know that underlying code is deprecated."""
    raise ImportError(
        "This agent has been moved to langchain experiment. "
        "This agent relies on python REPL tool under the hood, so to use it "
        "safely please sandbox the python REPL. "
        "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
        "To keep using this code as is, install langchain experimental and "
        "update relevant imports replacing 'langchain' with 'langchain_experimental'"
    )


raise_on_import()
