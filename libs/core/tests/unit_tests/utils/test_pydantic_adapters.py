"""Internal adapters for migrating from pydantic v1 to v2."""
from typing import Any

import pytest

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils._pydantic_adapters import get_model_defaults


class MyModel(BaseModel):
    field_with_default_int: int = Field(10)
    field_with_default_str: str = Field("default")
    field_with_default_none: Any = Field(None)
    field_without_default: float
    field_with_default_bool: bool = Field(True)


def test_get_model_defaults():
    # Test case 1: Get defaults for all fields with defaults
    model_class = MyModel
    fields = [
        "field_with_default_int",
        "field_with_default_str",
        "field_with_default_none",
        "field_with_default_bool",
    ]
    expected_defaults = [10, "default", None, True]
    assert get_model_defaults(model_class, fields) == expected_defaults

    # Test case 2: Get defaults for a subset of fields
    fields = ["field_with_default_str", "field_with_default_none"]
    expected_defaults = ["default", None]
    assert get_model_defaults(model_class, fields) == expected_defaults

    # Test case 3: Field without default should raise an AttributeError
    fields = ["field_without_default"]
    with pytest.raises(KeyError):
        get_model_defaults(model_class, fields)

    # Test case 4: Field with non-literal default value should raise NotImplementedError
    class TestModelWithNonLiteralDefault(BaseModel):
        field_with_non_literal_default: Any = Field([1, 2, 3])

    fields = ["field_with_non_literal_default"]
    with pytest.raises(NotImplementedError):
        get_model_defaults(TestModelWithNonLiteralDefault, fields)
