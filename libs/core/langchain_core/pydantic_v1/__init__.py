def _raise_import_error() -> None:
    """Raise ImportError with a helpful message."""
    raise ImportError(
        "Please do not import from langchain_core.pydantic_v1. "
        "This module was a compatibility shim for pydantic v1, and should "
        "no longer be used "
        "Please update the code to import from Pydantic directly. "
        "For example, replace imports like: "
        "`from langchain_core.pydantic_v1 import BaseModel`\n"
        "with: `from pydantic import BaseModel`\n"
    )


_raise_import_error()
