from langchain_core.utils.pydantic import PYDANTIC_VERSION


def get_pydantic_major_version() -> int:
    """Get the major version of Pydantic.

    Returns:
        The major version of Pydantic.
    """
    return PYDANTIC_VERSION.major


def _parse_obj(pydantic_object, obj):
    """Validate obj against pydantic model, allowing coercion."""
    return pydantic_object.model_validate(obj, strict=False)


__all__ = ["get_pydantic_major_version", "_parse_obj"]
