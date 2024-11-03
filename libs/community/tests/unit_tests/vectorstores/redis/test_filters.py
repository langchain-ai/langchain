from typing import Any

import pytest

from langchain_community.vectorstores.redis import (
    RedisNum as Num,
)
from langchain_community.vectorstores.redis import (
    RedisTag as Tag,
)
from langchain_community.vectorstores.redis import (
    RedisText as Text,
)


# Test cases for various tag scenarios
@pytest.mark.parametrize(
    "operation,tags,expected",
    [
        # Testing single tags
        ("==", "simpletag", "@tag_field:{simpletag}"),
        (
            "==",
            "tag with space",
            "@tag_field:{tag\\ with\\ space}",
        ),  # Escaping spaces within quotes
        (
            "==",
            "special$char",
            "@tag_field:{special\\$char}",
        ),  # Escaping a special character
        ("!=", "negated", "(-@tag_field:{negated})"),
        # Testing multiple tags
        ("==", ["tag1", "tag2"], "@tag_field:{tag1|tag2}"),
        (
            "==",
            ["alpha", "beta with space", "gamma$special"],
            "@tag_field:{alpha|beta\\ with\\ space|gamma\\$special}",
        ),  # Multiple tags with spaces and special chars
        ("!=", ["tagA", "tagB"], "(-@tag_field:{tagA|tagB})"),
        # Complex tag scenarios with special characters
        ("==", "weird:tag", "@tag_field:{weird\\:tag}"),  # Tags with colon
        ("==", "tag&another", "@tag_field:{tag\\&another}"),  # Tags with ampersand
        # Escaping various special characters within tags
        ("==", "tag/with/slashes", "@tag_field:{tag\\/with\\/slashes}"),
        (
            "==",
            ["hyphen-tag", "under_score", "dot.tag"],
            "@tag_field:{hyphen\\-tag|under_score|dot\\.tag}",
        ),
        # ...additional unique cases as desired...
    ],
)
def test_tag_filter_varied(operation: str, tags: str, expected: str) -> None:
    if operation == "==":
        tf = Tag("tag_field") == tags
    elif operation == "!=":
        tf = Tag("tag_field") != tags
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    # Verify the string representation matches the expected RediSearch query part
    assert str(tf) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "*"),
        ([], "*"),
        ("", "*"),
        ([None], "*"),
        ([None, "tag"], "@tag_field:{tag}"),
    ],
    ids=[
        "none",
        "empty_list",
        "empty_string",
        "list_with_none",
        "list_with_none_and_tag",
    ],
)
def test_nullable_tags(value: Any, expected: str) -> None:
    tag = Tag("tag_field")
    assert str(tag == value) == expected


@pytest.mark.parametrize(
    "operation, value, expected",
    [
        ("__eq__", 5, "@numeric_field:[5 5]"),
        ("__ne__", 5, "(-@numeric_field:[5 5])"),
        ("__gt__", 5, "@numeric_field:[(5 +inf]"),
        ("__ge__", 5, "@numeric_field:[5 +inf]"),
        ("__lt__", 5.55, "@numeric_field:[-inf (5.55]"),
        ("__le__", 5, "@numeric_field:[-inf 5]"),
        ("__le__", None, "*"),
        ("__eq__", None, "*"),
        ("__ne__", None, "*"),
    ],
    ids=["eq", "ne", "gt", "ge", "lt", "le", "le_none", "eq_none", "ne_none"],
)
def test_numeric_filter(operation: str, value: Any, expected: str) -> None:
    nf = Num("numeric_field")
    assert str(getattr(nf, operation)(value)) == expected


@pytest.mark.parametrize(
    "operation, value, expected",
    [
        ("__eq__", "text", '@text_field:("text")'),
        ("__ne__", "text", '(-@text_field:"text")'),
        ("__eq__", "", "*"),
        ("__ne__", "", "*"),
        ("__eq__", None, "*"),
        ("__ne__", None, "*"),
        ("__mod__", "text", "@text_field:(text)"),
        ("__mod__", "tex*", "@text_field:(tex*)"),
        ("__mod__", "%text%", "@text_field:(%text%)"),
        ("__mod__", "", "*"),
        ("__mod__", None, "*"),
    ],
    ids=[
        "eq",
        "ne",
        "eq-empty",
        "ne-empty",
        "eq-none",
        "ne-none",
        "like",
        "like_wildcard",
        "like_full",
        "like_empty",
        "like_none",
    ],
)
def test_text_filter(operation: str, value: Any, expected: str) -> None:
    txt_f = getattr(Text("text_field"), operation)(value)
    assert str(txt_f) == expected


def test_filters_combination() -> None:
    tf1 = Tag("tag_field") == ["tag1", "tag2"]
    tf2 = Tag("tag_field") == "tag3"
    combined = tf1 & tf2
    assert str(combined) == "(@tag_field:{tag1|tag2} @tag_field:{tag3})"

    combined = tf1 | tf2
    assert str(combined) == "(@tag_field:{tag1|tag2} | @tag_field:{tag3})"

    tf1 = Tag("tag_field") == []
    assert str(tf1) == "*"
    assert str(tf1 & tf2) == str(tf2)
    assert str(tf1 | tf2) == str(tf2)

    # test combining filters with None values and empty strings
    tf1 = Tag("tag_field") == None  # noqa: E711
    tf2 = Tag("tag_field") == ""
    assert str(tf1 & tf2) == "*"

    tf1 = Tag("tag_field") == None  # noqa: E711
    tf2 = Tag("tag_field") == "tag"
    assert str(tf1 & tf2) == str(tf2)

    tf1 = Tag("tag_field") == None  # noqa: E711
    tf2 = Tag("tag_field") == ["tag1", "tag2"]
    assert str(tf1 & tf2) == str(tf2)

    tf1 = Tag("tag_field") == None  # noqa: E711
    tf2 = Tag("tag_field") != None  # noqa: E711
    assert str(tf1 & tf2) == "*"

    tf1 = Tag("tag_field") == ""
    tf2 = Tag("tag_field") == "tag"
    tf3 = Tag("tag_field") == ["tag1", "tag2"]
    assert str(tf1 & tf2 & tf3) == str(tf2 & tf3)

    # test none filters for Tag Num Text
    tf1 = Tag("tag_field") == None  # noqa: E711
    tf2 = Num("num_field") == None  # noqa: E711
    tf3 = Text("text_field") == None  # noqa: E711
    assert str(tf1 & tf2 & tf3) == "*"

    tf1 = Tag("tag_field") != None  # noqa: E711
    tf2 = Num("num_field") != None  # noqa: E711
    tf3 = Text("text_field") != None  # noqa: E711
    assert str(tf1 & tf2 & tf3) == "*"

    # test combinations of real and None filters
    tf1 = Tag("tag_field") == "tag"
    tf2 = Num("num_field") == None  # noqa: E711
    tf3 = Text("text_field") == None  # noqa: E711
    assert str(tf1 & tf2 & tf3) == str(tf1)
