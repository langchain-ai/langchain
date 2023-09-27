import typing
from typing import Optional

import pytest
from langchain.load.dump import dumpd
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict

try:
    from pydantic.v1 import BaseModel, ValidationError
except ImportError:
    from pydantic import BaseModel, ValidationError

from langserve.validation import (
    create_batch_request_model,
    create_invoke_request_model,
    create_runnable_config_model,
    replace_lc_object_types,
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


def test_replace_lc_object_types() -> None:
    """Replace lc object types in a model."""
    updated_type = replace_lc_object_types(typing.List[HumanMessage])
    config = create_runnable_config_model("test", [])
    invoke_request = create_invoke_request_model("namespace", updated_type, config)
    invoke_request(
        input=dumpd(
            [
                HumanMessage(content="Hello, world!"),
                HumanMessage(content="Hello, world 2!"),
            ]
        )
    )

    with pytest.raises(ValidationError):
        invoke_request(input=[dumpd(AIMessage(content="Hello, world!"))])

    with pytest.raises(ValidationError):
        invoke_request(
            input=dumpd(
                [
                    AIMessage(content="Hello, world!"),
                    HumanMessage(content="Hello, world!"),
                ]
            ),
        )


def test_batch_request_with_lc_serialization() -> None:
    """Test batch request with LC serialization."""

    input_type = replace_lc_object_types(typing.List[HumanMessage])
    config = create_runnable_config_model("test", [])
    batch_request = create_batch_request_model("namespace", input_type, config)
    with pytest.raises(ValidationError):
        batch_request(inputs=dumpd([[SystemMessage(content="Hello, world!")]]))

    with pytest.raises(ValidationError):
        batch_request(inputs=dumpd(HumanMessage(content="Hello, world!")))

    with pytest.raises(ValidationError):
        batch_request(inputs=dumpd([HumanMessage(content="Hello, world!")]))

    batch_request(inputs=dumpd([[HumanMessage(content="Hello, world!")]]))


class PlaceHolderTypedDict(TypedDict):
    x: int
    z: HumanMessage


@pytest.mark.parametrize(
    "type_,input,is_valid",
    [
        (None, None, True),
        (str, "hello", True),
        (str, 123.0, True),
        (float, "qwe", False),
        (int, 1, True),
        (int, "qwe", False),
        (typing.Union[str, int], "hello", True),
        (typing.Union[str, int], 3, True),
        (typing.List[str], ["a", "b"], True),
        (typing.List[str], ["a", None], False),
        (typing.List[HumanMessage], [HumanMessage(content="hello, world!")], True),
        (typing.List[HumanMessage], [SystemMessage(content="hello, world!")], False),
        (
            typing.List[typing.Union[HumanMessage, SystemMessage]],
            [HumanMessage(content="he"), SystemMessage(content="hello, world!")],
            True,
        ),
        (
            typing.List[typing.Union[HumanMessage, SystemMessage]],
            HumanMessage(content="hello"),
            False,
        ),
        (
            typing.Union[
                typing.List[typing.Union[SystemMessage, HumanMessage, str]], str
            ],
            ["hello", "world"],
            True,
        ),
    ],
)
def test_replace_lc_object_type(
    type_: typing.Any, input: typing.Any, is_valid: bool
) -> None:
    """Verify that code runs on different python versions."""
    new_type = replace_lc_object_types(type_)

    class Model(BaseModel):
        input_: new_type

    if is_valid:
        Model(input_=dumpd(input))
    else:
        with pytest.raises(ValidationError):
            Model(input_=dumpd(input))
