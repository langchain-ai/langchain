import sys
from typing import Callable

import pytest

from langchain.schema.runnable.utils import (
    get_lambda_source,
    indent_lines_after_first,
)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
@pytest.mark.parametrize(
    "func, expected_source",
    [
        (lambda x: x * 2, "lambda x: x * 2"),
        (lambda a, b: a + b, "lambda a, b: a + b"),
        (lambda x: x if x > 0 else 0, "lambda x: x if x > 0 else 0"),
    ],
)
def test_get_lambda_source(func: Callable, expected_source: str) -> None:
    """Test get_lambda_source function"""
    source = get_lambda_source(func)
    assert source == expected_source


@pytest.mark.parametrize(
    "text,prefix,expected_output",
    [
        ("line 1\nline 2\nline 3", "1", "line 1\n line 2\n line 3"),
        ("line 1\nline 2\nline 3", "ax", "line 1\n  line 2\n  line 3"),
    ],
)
def test_indent_lines_after_first(text: str, prefix: str, expected_output: str) -> None:
    """Test indent_lines_after_first function"""
    indented_text = indent_lines_after_first(text, prefix)
    assert indented_text == expected_output
