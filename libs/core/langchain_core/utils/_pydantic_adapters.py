"""Internal adapters for migrating from pydantic v1 to v2."""
from typing import Any, List, Sequence, Type

from langchain_core.pydantic_v1 import BaseModel


def get_model_defaults(
    model_class: Type[BaseModel], fields: Sequence[str]
) -> List[Any]:
    """Get model defaults for the given fields.

    Args:
        model_class: The model class.
        fields: The fields to get defaults for.

    Returns:
        The defaults for the given fields.

        Only supports default literals right now
    """
    values = []
    for field in fields:
        model_field = model_class.__fields__[field]
        if model_field.required:
            raise KeyError(f"Field {field} is required")

        default_value = model_class.__fields__[field].default

        if not isinstance(default_value, (int, float, str, bool, type(None))):
            raise NotImplementedError("Only literals are supported for now")

        values.append(default_value)

    return values
