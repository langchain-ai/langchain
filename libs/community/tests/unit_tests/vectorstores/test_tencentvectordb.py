import pytest

from langchain_community.vectorstores.tencentvectordb import TencentVectorDB, translate_filter


def test_translate_filter():
    raw_filter = "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180))"
    result = translate_filter(raw_filter)
    expr = "(artist = \"Taylor Swift\" or artist = \"Katy Perry\") and length < 180"
    assert expr == result


def test_translate_filter_with_in_comparison():
    raw_filter = "in(\"artist\", [\"Taylor Swift\", \"Katy Perry\"])"
    result = translate_filter(raw_filter)
    expr = "artist in (\"Taylor Swift\", \"Katy Perry\")"
    assert expr == result

