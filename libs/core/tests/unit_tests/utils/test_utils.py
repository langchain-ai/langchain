import os
import re
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from unittest.mock import patch

import pytest

from langchain_core import utils
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.utils import (
    check_package_version,
    from_env,
    get_pydantic_field_names,
    guard_import,
)
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION
from langchain_core.utils.utils import secret_from_env


@pytest.mark.parametrize(
    ("package", "check_kwargs", "actual_version", "expected"),
    [
        ("stub", {"gt_version": "0.1"}, "0.1.2", None),
        ("stub", {"gt_version": "0.1.2"}, "0.1.12", None),
        ("stub", {"gt_version": "0.1.2"}, "0.1.2", (ValueError, "> 0.1.2")),
        ("stub", {"gte_version": "0.1"}, "0.1.2", None),
        ("stub", {"gte_version": "0.1.2"}, "0.1.2", None),
    ],
)
def test_check_package_version(
    package: str,
    check_kwargs: Dict[str, Optional[str]],
    actual_version: str,
    expected: Optional[Tuple[Type[Exception], str]],
) -> None:
    with patch("langchain_core.utils.utils.version", return_value=actual_version):
        if expected is None:
            check_package_version(package, **check_kwargs)
        else:
            with pytest.raises(expected[0], match=expected[1]):
                check_package_version(package, **check_kwargs)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        # Merge `None` and `1`.
        ({"a": None}, {"a": 1}, {"a": 1}),
        # Merge `1` and `None`.
        ({"a": 1}, {"a": None}, {"a": 1}),
        # Merge `None` and a value.
        ({"a": None}, {"a": 0}, {"a": 0}),
        ({"a": None}, {"a": "txt"}, {"a": "txt"}),
        # Merge equal values.
        ({"a": 1}, {"a": 1}, {"a": 1}),
        ({"a": 1.5}, {"a": 1.5}, {"a": 1.5}),
        ({"a": True}, {"a": True}, {"a": True}),
        ({"a": False}, {"a": False}, {"a": False}),
        ({"a": "txt"}, {"a": "txt"}, {"a": "txttxt"}),
        ({"a": [1, 2]}, {"a": [1, 2]}, {"a": [1, 2, 1, 2]}),
        ({"a": {"b": "txt"}}, {"a": {"b": "txt"}}, {"a": {"b": "txttxt"}}),
        # Merge strings.
        ({"a": "one"}, {"a": "two"}, {"a": "onetwo"}),
        # Merge dicts.
        ({"a": {"b": 1}}, {"a": {"c": 2}}, {"a": {"b": 1, "c": 2}}),
        (
            {"function_call": {"arguments": None}},
            {"function_call": {"arguments": "{\n"}},
            {"function_call": {"arguments": "{\n"}},
        ),
        # Merge lists.
        ({"a": [1, 2]}, {"a": [3]}, {"a": [1, 2, 3]}),
        ({"a": 1, "b": 2}, {"a": 1}, {"a": 1, "b": 2}),
        ({"a": 1, "b": 2}, {"c": None}, {"a": 1, "b": 2, "c": None}),
        #
        # Invalid inputs.
        #
        (
            {"a": 1},
            {"a": "1"},
            pytest.raises(
                TypeError,
                match=re.escape(
                    'additional_kwargs["a"] already exists in this message, '
                    "but with a different type."
                ),
            ),
        ),
        (
            {"a": (1, 2)},
            {"a": (3,)},
            pytest.raises(
                TypeError,
                match=(
                    "Additional kwargs key a already exists in left dict and value "
                    "has unsupported type .+tuple.+."
                ),
            ),
        ),
        # 'index' keyword has special handling
        (
            {"a": [{"index": 0, "b": "{"}]},
            {"a": [{"index": 0, "b": "f"}]},
            {"a": [{"index": 0, "b": "{f"}]},
        ),
        (
            {"a": [{"idx": 0, "b": "{"}]},
            {"a": [{"idx": 0, "b": "f"}]},
            {"a": [{"idx": 0, "b": "{"}, {"idx": 0, "b": "f"}]},
        ),
    ),
)
def test_merge_dicts(
    left: dict, right: dict, expected: Union[dict, AbstractContextManager]
) -> None:
    if isinstance(expected, AbstractContextManager):
        err = expected
    else:
        err = nullcontext()

    left_copy = deepcopy(left)
    right_copy = deepcopy(right)
    with err:
        actual = merge_dicts(left, right)
        assert actual == expected
        # no mutation
        assert left == left_copy
        assert right == right_copy


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        # 'type' special key handling
        ({"type": "foo"}, {"type": "foo"}, {"type": "foo"}),
        (
            {"type": "foo"},
            {"type": "bar"},
            pytest.raises(ValueError, match="Unable to merge."),
        ),
    ),
)
@pytest.mark.xfail(reason="Refactors to make in 0.3")
def test_merge_dicts_0_3(
    left: dict, right: dict, expected: Union[dict, AbstractContextManager]
) -> None:
    if isinstance(expected, AbstractContextManager):
        err = expected
    else:
        err = nullcontext()

    left_copy = deepcopy(left)
    right_copy = deepcopy(right)
    with err:
        actual = merge_dicts(left, right)
        assert actual == expected
        # no mutation
        assert left == left_copy
        assert right == right_copy


@pytest.mark.parametrize(
    ("module_name", "pip_name", "package", "expected"),
    [
        ("langchain_core.utils", None, None, utils),
        ("langchain_core.utils", "langchain-core", None, utils),
        ("langchain_core.utils", None, "langchain-core", utils),
        ("langchain_core.utils", "langchain-core", "langchain-core", utils),
    ],
)
def test_guard_import(
    module_name: str, pip_name: Optional[str], package: Optional[str], expected: Any
) -> None:
    if package is None and pip_name is None:
        ret = guard_import(module_name)
    elif package is None and pip_name is not None:
        ret = guard_import(module_name, pip_name=pip_name)
    elif package is not None and pip_name is None:
        ret = guard_import(module_name, package=package)
    elif package is not None and pip_name is not None:
        ret = guard_import(module_name, pip_name=pip_name, package=package)
    else:
        raise ValueError("Invalid test case")
    assert ret == expected


@pytest.mark.parametrize(
    ("module_name", "pip_name", "package"),
    [
        ("langchain_core.utilsW", None, None),
        ("langchain_core.utilsW", "langchain-core-2", None),
        ("langchain_core.utilsW", None, "langchain-coreWX"),
        ("langchain_core.utilsW", "langchain-core-2", "langchain-coreWX"),
        ("langchain_coreW", None, None),  # ModuleNotFoundError
    ],
)
def test_guard_import_failure(
    module_name: str, pip_name: Optional[str], package: Optional[str]
) -> None:
    with pytest.raises(ImportError) as exc_info:
        if package is None and pip_name is None:
            guard_import(module_name)
        elif package is None and pip_name is not None:
            guard_import(module_name, pip_name=pip_name)
        elif package is not None and pip_name is None:
            guard_import(module_name, package=package)
        elif package is not None and pip_name is not None:
            guard_import(module_name, pip_name=pip_name, package=package)
        else:
            raise ValueError("Invalid test case")
    pip_name = pip_name or module_name.split(".")[0].replace("_", "-")
    err_msg = (
        f"Could not import {module_name} python package. "
        f"Please install it with `pip install {pip_name}`."
    )
    assert exc_info.value.msg == err_msg


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Requires pydantic 2")
def test_get_pydantic_field_names_v1_in_2() -> None:
    from pydantic.v1 import BaseModel as PydanticV1BaseModel  # pydantic: ignore
    from pydantic.v1 import Field  # pydantic: ignore

    class PydanticV1Model(PydanticV1BaseModel):
        field1: str
        field2: int
        alias_field: int = Field(alias="aliased_field")

    result = get_pydantic_field_names(PydanticV1Model)
    expected = {"field1", "field2", "aliased_field", "alias_field"}
    assert result == expected


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Requires pydantic 2")
def test_get_pydantic_field_names_v2_in_2() -> None:
    from pydantic import BaseModel, Field  # pydantic: ignore

    class PydanticModel(BaseModel):
        field1: str
        field2: int
        alias_field: int = Field(alias="aliased_field")

    result = get_pydantic_field_names(PydanticModel)
    expected = {"field1", "field2", "aliased_field", "alias_field"}
    assert result == expected


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 1, reason="Requires pydantic 1")
def test_get_pydantic_field_names_v1() -> None:
    from pydantic import BaseModel, Field  # pydantic: ignore

    class PydanticModel(BaseModel):
        field1: str
        field2: int
        alias_field: int = Field(alias="aliased_field")

    result = get_pydantic_field_names(PydanticModel)
    expected = {"field1", "field2", "aliased_field", "alias_field"}
    assert result == expected


def test_from_env_with_env_variable() -> None:
    key = "TEST_KEY"
    value = "test_value"
    with patch.dict(os.environ, {key: value}):
        get_value = from_env(key)
        assert get_value() == value


def test_from_env_with_default_value() -> None:
    key = "TEST_KEY"
    default_value = "default_value"
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(key, default=default_value)
        assert get_value() == default_value


def test_from_env_with_error_message() -> None:
    key = "TEST_KEY"
    error_message = "Custom error message"
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(key, error_message=error_message)
        with pytest.raises(ValueError, match=error_message):
            get_value()


def test_from_env_with_default_error_message() -> None:
    key = "TEST_KEY"
    with patch.dict(os.environ, {}, clear=True):
        get_value = from_env(key)
        with pytest.raises(ValueError, match=f"Did not find {key}"):
            get_value()


def test_secret_from_env_with_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set the environment variable
    monkeypatch.setenv("TEST_KEY", "secret_value")

    # Get the function
    get_secret: Callable[[], Optional[SecretStr]] = secret_from_env("TEST_KEY")

    # Assert that it returns the correct value
    assert get_secret() == SecretStr("secret_value")


def test_secret_from_env_with_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function with a default value
    get_secret: Callable[[], SecretStr] = secret_from_env(
        "TEST_KEY", default="default_value"
    )

    # Assert that it returns the default value
    assert get_secret() == SecretStr("default_value")


def test_secret_from_env_with_none_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function with a default value of None
    get_secret: Callable[[], Optional[SecretStr]] = secret_from_env(
        "TEST_KEY", default=None
    )

    # Assert that it returns None
    assert get_secret() is None


def test_secret_from_env_without_default_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function without a default value
    get_secret: Callable[[], SecretStr] = secret_from_env("TEST_KEY")

    # Assert that it raises a ValueError with the correct message
    with pytest.raises(ValueError, match="Did not find TEST_KEY"):
        get_secret()


def test_secret_from_env_with_custom_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset the environment variable
    monkeypatch.delenv("TEST_KEY", raising=False)

    # Get the function without a default value but with a custom error message
    get_secret: Callable[[], SecretStr] = secret_from_env(
        "TEST_KEY", error_message="Custom error message"
    )

    # Assert that it raises a ValueError with the custom message
    with pytest.raises(ValueError, match="Custom error message"):
        get_secret()


def test_using_secret_from_env_as_default_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.pydantic_v1 import BaseModel, Field

    class Foo(BaseModel):
        secret: SecretStr = Field(default_factory=secret_from_env("TEST_KEY"))

    # Pass the secret as a parameter
    foo = Foo(secret="super_secret")  # type: ignore[arg-type]
    assert foo.secret.get_secret_value() == "super_secret"

    # Set the environment variable
    monkeypatch.setenv("TEST_KEY", "secret_value")
    assert Foo().secret.get_secret_value() == "secret_value"

    class Bar(BaseModel):
        secret: Optional[SecretStr] = Field(
            default_factory=secret_from_env("TEST_KEY_2", default=None)
        )

    assert Bar().secret is None

    class Buzz(BaseModel):
        secret: Optional[SecretStr] = Field(
            default_factory=secret_from_env("TEST_KEY_2", default="hello")
        )

    # We know it will be SecretStr rather than Optional[SecretStr]
    assert Buzz().secret.get_secret_value() == "hello"  # type: ignore

    class OhMy(BaseModel):
        secret: Optional[SecretStr] = Field(
            default_factory=secret_from_env("FOOFOOFOOBAR")
        )

    with pytest.raises(ValueError, match="Did not find FOOFOOFOOBAR"):
        OhMy()
