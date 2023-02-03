import pytest

from langchain.utilities.normalization import normalize_boolean_output

import re
import pytest

def test_normalize_boolean_output():
    def run_test(input_string, expected_output, true_values=["1"], false_values=["0"]):
        assert normalize_boolean_output(input_string, true_values, false_values) == expected_output
    
    run_test("0", False)
    run_test("1", True)
    run_test("\n1\n", True)
    run_test("The answer is: \n1\n", True)
    run_test("The answer is: 0", False)
    run_test("1", False, ["0"], ["1"])
    run_test("0", True, ["0"], ["1"])
    run_test("X", True, ["x", "X"], ["O", "o"])
    with pytest.raises(ValueError):
        normalize_boolean_output("1", ["0", "1"], ["0", "1"])
    with pytest.raises(ValueError):
        normalize_boolean_output("01", ["1"], ["0"])
    with pytest.raises(ValueError):
        normalize_boolean_output("", ["1"], ["0"])
    with pytest.raises(ValueError):
        normalize_boolean_output("a", ["0"], ["1"])
    with pytest.raises(ValueError):
        normalize_boolean_output("2", ["1"], ["0"])
