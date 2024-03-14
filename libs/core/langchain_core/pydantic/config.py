import os


def _get_use_pydantic_v2() -> bool:
    """Get the value of the LC_PYDANTIC_V2_UNSAFE environment variable."""
    value = os.environ.get("LC_PYDANTIC_V2_UNSAFE", "false").lower()
    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        raise ValueError(f"Invalid value for LANGCHAIN_PYDANTIC_V2_UNSAFE: {value}")


USE_PYDANTIC_V2 = _get_use_pydantic_v2()
