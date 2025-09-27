from langchain_core.utils.pydantic import PYDANTIC_VERSION


def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic.

    Returns:
        The major version of Pydantic.
    """
    return PYDANTIC_VERSION.major


__all__ = ["get_pydantic_major_version"]
