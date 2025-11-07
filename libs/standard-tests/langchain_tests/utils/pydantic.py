"""Utilities for working with pydantic models."""


def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic."""
    try:
        import pydantic  # noqa: PLC0415

        return int(pydantic.__version__.split(".")[0])
    except ImportError:
        return 0


PYDANTIC_MAJOR_VERSION = get_pydantic_major_version()
