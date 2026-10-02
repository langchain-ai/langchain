from typing import Any


def __getattr__(_: str = "") -> Any:
    msg = (
        "This tool has been moved to langchain_experimental. "
        "This tool has access to a python REPL. "
        "For best practices make sure to sandbox this tool. "
        "Read https://docs.langchain.com/oss/python/security-policy. "
        "To keep using this code as is, install langchain_experimental and "
        "update relevant imports replacing 'langchain' with 'langchain_experimental'"
    )
    raise AttributeError(msg)
