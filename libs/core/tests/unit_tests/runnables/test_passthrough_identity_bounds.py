\"\"\"
Unit tests for RunnablePassthrough input verification and identity mapping edge cases.
\"\"\"
import pytest
from langchain_core.runnables import RunnablePassthrough

def test_runnable_passthrough_scalar_identity():
    passthrough = RunnablePassthrough()
    assert passthrough.invoke(42) == 42
    assert passthrough.invoke("test-string") == "test-string"
    assert passthrough.invoke(None) is None

def test_runnable_passthrough_dict_identity():
    passthrough = RunnablePassthrough()
    input_dict = {"query": "hello", "count": 5}
    output_dict = passthrough.invoke(input_dict)
    assert output_dict == input_dict
    assert output_dict["query"] == "hello"

def test_runnable_passthrough_assign_key():
    passthrough = RunnablePassthrough.assign(extra=lambda x: x["val"] * 2)
    result = passthrough.invoke({"val": 10})
    assert result == {"val": 10, "extra": 20}