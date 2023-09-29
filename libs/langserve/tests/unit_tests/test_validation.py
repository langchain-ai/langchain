from typing import Optional

import pytest

try:
    from pydantic.v1 import BaseModel, ValidationError
except ImportError:
    from pydantic import BaseModel, ValidationError

from langserve.validation import (
    create_batch_request_model,
    create_invoke_request_model,
    create_runnable_config_model,
)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "input": {"a": "qqq"},
            "kwargs": {},
            "valid": False,
        },
        {
            "input": {"a": 2},
            "kwargs": "hello",
            "valid": False,
        },
        {
            "input": {"a": 2},
            "config": "hello",
            "valid": False,
        },
        {
            "input": {"b": "hello"},
            "valid": False,
        },
        {
            "input": {"a": 2, "b": "hello"},
            "config": "hello",
            "valid": False,
        },
        {
            "input": {"a": 2, "b": "hello"},
            "valid": True,
        },
        {
            "input": {"a": 2, "b": "hello"},
            "valid": True,
        },
        {
            "input": {"a": 2},
            "valid": True,
        },
    ],
)
def test_create_invoke_and_batch_models(test_case: dict) -> None:
    """Test that the invoke request model is created correctly."""

    class Input(BaseModel):
        """Test input."""

        a: int
        b: Optional[str] = None

    valid = test_case.pop("valid")
    config = create_runnable_config_model("test", ["tags"])

    model = create_invoke_request_model("namespace", Input, config)

    if valid:
        model(**test_case)
    else:
        with pytest.raises(ValidationError):
            model(**test_case)

    # Validate batch request
    # same structure as input request, but
    # 'input' is a list of inputs and is called 'inputs'
    batch_model = create_batch_request_model("namespace", Input, config)

    test_case["inputs"] = [test_case.pop("input")]
    if valid:
        batch_model(**test_case)
    else:
        with pytest.raises(ValidationError):
            batch_model(**test_case)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "type": int,
            "input": 1,
            "valid": True,
        },
        {
            "type": float,
            "input": "name",
            "valid": False,
        },
        {
            "type": float,
            "input": [3.2],
            "valid": False,
        },
        {
            "type": float,
            "input": 1.1,
            "valid": True,
        },
        {
            "type": Optional[float],
            "valid": True,
            "input": None,
        },
    ],
)
def test_validation(test_case) -> None:
    """Test that the invoke request model is created correctly."""
    config = create_runnable_config_model("test", [])
    model = create_invoke_request_model("namespace", test_case.pop("type"), config)

    if test_case["valid"]:
        model(**test_case)
    else:
        with pytest.raises(ValidationError):
            model(**test_case)
