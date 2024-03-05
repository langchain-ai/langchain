import re
from contextlib import AbstractContextManager, nullcontext
from typing import Dict, Optional, Tuple, Type, Union
from unittest.mock import patch

import pytest

from langchain_core.utils import check_package_version
from langchain_core.utils._merge import merge_dicts


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
        ({"a": "txt"}, {"a": "txt"}, {"a": "txt"}),
        ({"a": [1, 2]}, {"a": [1, 2]}, {"a": [1, 2]}),
        ({"a": {"b": "txt"}}, {"a": {"b": "txt"}}, {"a": {"b": "txt"}}),
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
    ),
)
def test_merge_dicts(
    left: dict, right: dict, expected: Union[dict, AbstractContextManager]
) -> None:
    if isinstance(expected, AbstractContextManager):
        err = expected
    else:
        err = nullcontext()

    with err:
        actual = merge_dicts(left, right)
        assert actual == expected
