\"\"\"
Unit tests for RunnableLambda execution boundaries, typing, and exception propagation in langchain_core.runnables.
\"\"\"
import pytest
from langchain_core.runnables import RunnableLambda

def test_runnable_lambda_synchronous_execution():
    def multiply_by_three(x: int) -> int:
        return x * 3

    runnable = RunnableLambda(multiply_by_three)
    assert runnable.invoke(7) == 21

def test_runnable_lambda_with_kwargs():
    def add_offset(x: int, offset: int = 10) -> int:
        return x + offset

    runnable = RunnableLambda(add_offset)
    assert runnable.invoke(5) == 15

def test_runnable_lambda_exception_propagation():
    def faulty_function(x: int) -> int:
        if x < 0:
            raise ValueError("Negative values not permitted")
        return x

    runnable = RunnableLambda(faulty_function)
    with pytest.raises(ValueError, match="Negative values not permitted"):
        runnable.invoke(-1)