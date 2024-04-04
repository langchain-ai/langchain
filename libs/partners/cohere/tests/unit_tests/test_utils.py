import pytest

from langchain_cohere.utils import _remove_signature_from_tool_description


@pytest.mark.parametrize(
    "name,description,expected",
    [
        pytest.param(
            "foo", "bar baz", "bar baz", id="description doesn't have signature"
        ),
        pytest.param("foo", "", "", id="description is empty"),
        pytest.param("foo", "foo(a: str) - bar baz", "bar baz", id="signature"),
        pytest.param(
            "foo", "foo() - bar baz", "bar baz", id="signature with empty args"
        ),
        pytest.param(
            "foo",
            "foo(a: str) - foo(b: str) - bar",
            "foo(b: str) - bar",
            id="signature with edge case",
        ),
        pytest.param(
            "foo", "foo() -> None - bar baz", "bar baz", id="signature with return type"
        ),
        pytest.param(
            "foo",
            """My description.

Args:
    Bar: 
""",
            "My description.",
            id="signature with Args: section",
        ),
    ],
)
def test_remove_signature_from_description(
    name: str, description: str, expected: str
) -> None:
    actual = _remove_signature_from_tool_description(name=name, description=description)

    assert expected == actual
