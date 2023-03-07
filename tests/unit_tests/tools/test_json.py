"""Test functionality of JSON tools."""
from pathlib import Path

from langchain.tools.json.tool import JsonSpec


def test_json_spec_from_file(tmp_path: Path) -> None:
    """Test JsonSpec can be constructed from a file."""
    path = tmp_path / "test.json"
    path.write_text('{"foo": "bar"}')
    spec = JsonSpec.from_file(path)
    assert spec.dict_ == {"foo": "bar"}


def test_json_spec_keys() -> None:
    """Test JsonSpec can return keys of a dict at given path."""
    spec = JsonSpec(dict_={"foo": "bar", "baz": {"test": {"foo": [1, 2, 3]}}})
    assert spec.keys("data") == "['foo', 'baz']"
    assert "ValueError" in spec.keys('data["foo"]')
    assert spec.keys('data["baz"]') == "['test']"
    assert spec.keys('data["baz"]["test"]') == "['foo']"
    assert "ValueError" in spec.keys('data["baz"]["test"]["foo"]')


def test_json_spec_value() -> None:
    """Test JsonSpec can return value of a dict at given path."""
    spec = JsonSpec(dict_={"foo": "bar", "baz": {"test": {"foo": [1, 2, 3]}}})
    assert spec.value("data") == "{'foo': 'bar', 'baz': {'test': {'foo': [1, 2, 3]}}}"
    assert spec.value('data["foo"]') == "bar"
    assert spec.value('data["baz"]') == "{'test': {'foo': [1, 2, 3]}}"
    assert spec.value('data["baz"]["test"]') == "{'foo': [1, 2, 3]}"
    assert spec.value('data["baz"]["test"]["foo"]') == "[1, 2, 3]"


def test_json_spec_value_max_length() -> None:
    """Test JsonSpec can return value of a dict at given path."""
    spec = JsonSpec(
        dict_={"foo": "bar", "baz": {"test": {"foo": [1, 2, 3]}}}, max_value_length=5
    )
    assert spec.value('data["foo"]') == "bar"
    assert (
        spec.value('data["baz"]')
        == "Value is a large dictionary, should explore its keys directly"
    )
    assert (
        spec.value('data["baz"]["test"]')
        == "Value is a large dictionary, should explore its keys directly"
    )
    assert spec.value('data["baz"]["test"]["foo"]') == "[1, 2..."
