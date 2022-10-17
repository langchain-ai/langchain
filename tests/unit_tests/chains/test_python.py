"""Test python chain."""

from langchain.chains.python import PythonChain


def test_functionality() -> None:
    """Test correct functionality."""
    chain = PythonChain(input_key="code1", output_key="output1")
    code = "print(1 + 1)"
    output = chain({"code1": code})
    assert output == {"code1": code, "output1": "2\n"}

    # Test with the more user-friendly interface.
    simple_output = chain.run(code)
    assert simple_output == "2\n"
