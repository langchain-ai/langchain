\"\"\"
Unit tests for string manipulation and template formatting edge cases in langchain_core.utils.strings.
\"\"\"
import pytest
from langchain_core.utils.strings import comma_list, stringify_value, stringify_dict

def test_comma_list_formatting():
    assert comma_list(["apple", "banana", "cherry"]) == "apple, banana, cherry"
    assert comma_list(["single"]) == "single"
    assert comma_list([]) == ""

def test_stringify_value_types():
    assert stringify_value("test") == "test"
    assert stringify_value(123) == "123"
    assert stringify_value(True) == "True"
    assert stringify_value(None) == "None"

def test_stringify_dict_formatting():
    d = {"k1": "v1", "k2": 2}
    result = stringify_dict(d)
    assert "k1: v1" in result
    assert "k2: 2" in result