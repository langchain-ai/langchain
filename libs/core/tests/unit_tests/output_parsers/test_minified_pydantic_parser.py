"""Unit tests for MinifiedPydanticOutputParser."""

import json
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from langchain_core.output_parsers.minified_pydantic import MinifiedPydanticOutputParser
from langchain_core.outputs.generation import Generation


class User(BaseModel):
    """A simple user model with minified field names."""

    first_name: str = Field(..., description="The user's first name")
    last_name: str = Field(..., description="The user's last name")
    email: str = Field(..., description="The user's email address")
    age: Optional[int] = Field(None, description="The user's age (optional)")


class Address(BaseModel):
    """A simple address model with minified field names."""

    street: str = Field(..., description="Street name")
    city: str = Field(..., description="City name")


class UserWithAddress(BaseModel):
    """A user model that includes a list of addresses."""

    user: User
    addresses: list[Address]


def test_minification_and_reverse_mapping() -> None:
    """Test that the minified parser correctly maps fields and descriptions."""

    parser = MinifiedPydanticOutputParser(pydantic_object=User)
    minified_cls = parser.minified
    # Check minified field names are short and descriptions are preserved
    fields = minified_cls.model_fields
    assert set(fields.keys()) == {"a", "b", "c", "d"}
    assert fields["a"].description == "The user's first name"
    assert fields["d"].description == "The user's age (optional)"

    # Prepare minified input (simulate LLM output)
    minified_dict = {"a": "John", "b": "Doe", "c": "john@example.com", "d": 42}
    minified_input = Generation(text=json.dumps(minified_dict))
    result = parser.parse_result([minified_input])
    assert isinstance(result, User)
    assert result.first_name == "John"
    assert result.last_name == "Doe"
    assert result.email == "john@example.com"
    assert result.age == 42


def test_optional_and_none_values() -> None:
    """Test that optional fields handle None values correctly."""
    parser = MinifiedPydanticOutputParser(User)

    minified_dict: dict[str, Union[str, None]] = {
        "a": "Jane",
        "b": "Smith",
        "c": "jane@example.com",
    }  # no 'd'
    minified_input = Generation(text=json.dumps(minified_dict))
    result = parser.parse_result([minified_input])
    assert isinstance(result, User), f"Expected User, got {type(result)}"
    assert result.age is None

    # Test with explicit None
    minified_dict["d"] = None
    minified_input = Generation(text=json.dumps(minified_dict))
    result = parser.parse_result([minified_input])
    assert isinstance(result, User), f"Expected User, got {type(result)}"
    assert result.age is None


def test_nested_model_and_list() -> None:
    """Test parsing a nested model with a list of addresses."""
    parser = MinifiedPydanticOutputParser(UserWithAddress)
    minified_cls = parser.minified
    # Should have two fields: user and addresses, both minified
    assert len(minified_cls.model_fields) == 2

    minified_dict = {
        "a": {"b": "Alice", "c": "Wonder", "d": "alice@ex.com", "e": None},  # user
        "f": [
            {"g": "123 Main", "h": "Metropolis"},
            {"g": "456 Side", "h": "Gotham"},
        ],
    }
    minified_input = Generation(text=json.dumps(minified_dict))
    result = parser.parse_result([minified_input])
    assert isinstance(result, UserWithAddress)
    assert result.user.first_name == "Alice"
    assert result.addresses[0].city == "Metropolis"
    assert result.addresses[1].street == "456 Side"


def test_remove_none_values() -> None:
    """Test that None values are removed from nested dicts/lists."""
    parser = MinifiedPydanticOutputParser(User)
    # Should remove None from nested dicts/lists
    data: dict[str, Union[str, None]] = {"a": "X", "b": "Y", "c": "Z", "d": None}
    cleaned = parser._remove_none_values(data)  # pylint: disable=protected-access
    assert isinstance(cleaned, dict), f"Expected dict, got {type(cleaned)}"
    assert "d" not in cleaned

    # Use Any for nested structures or be more specific about the type
    data_recurs: dict[str, Any] = {
        "a": None,
        "b": {"c": None, "d": 1},
        "e": [None, 2],
    }
    cleaned = parser._remove_none_values(  # pylint: disable=protected-access
        data_recurs
    )
    assert isinstance(cleaned, dict), f"Expected dict, got {type(cleaned)}"
    assert "a" not in cleaned
    assert cleaned["b"] == {"d": 1}
    assert cleaned["e"] == [2]


def test_strict() -> None:
    """Test the strict mode"""

    parser = MinifiedPydanticOutputParser(User, strict=True)
    minified_cls = parser.minified
    assert "required=True" in str(minified_cls.model_fields["a"])
    assert "required=True" in str(minified_cls.model_fields["b"])
    assert "required=True" in str(minified_cls.model_fields["c"])
    assert "required=True" in str(minified_cls.model_fields["d"])
