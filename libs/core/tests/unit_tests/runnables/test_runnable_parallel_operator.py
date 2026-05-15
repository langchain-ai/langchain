"""Unit tests for the & operator shorthand for RunnableParallel."""

import pytest
from langchain_core.runnables import RunnableLambda, RunnableParallel

double = RunnableLambda(lambda x: x * 2)
triple = RunnableLambda(lambda x: x * 3)
add_one = RunnableLambda(lambda x: x + 1)

def test_and_returns_runnable_parallel():
    result = double & triple
    assert isinstance(result, RunnableParallel)

def test_and_invoke_correct_output():
    parallel = double & triple
    output = parallel.invoke(5)
    assert output == {"left": 10, "right": 15}

def test_and_with_callable():
    parallel = double & (lambda x: x + 100)
    output = parallel.invoke(3)
    assert output == {"left": 6, "right": 103}

def test_rand_invoke_correct_output():
    parallel = double.__rand__(triple)
    output = parallel.invoke(5)
    assert output == {"left": 15, "right": 10}

def test_and_inside_pipe_sequence():
    chain = add_one | (double & triple)
    output = chain.invoke(4)
    assert output == {"left": 10, "right": 15}

def test_pipe_after_and():
    sum_values = RunnableLambda(lambda d: d["left"] + d["right"])
    chain = (double & triple) | sum_values
    output = chain.invoke(5)
    assert output == 25

@pytest.mark.asyncio
async def test_and_ainvoke():
    parallel = double & triple
    output = await parallel.ainvoke(5)
    assert output == {"left": 10, "right": 15}

def test_and_batch():
    parallel = double & triple
    outputs = parallel.batch([1, 2, 3])
    assert outputs == [
        {"left": 2, "right": 3},
        {"left": 4, "right": 6},
        {"left": 6, "right": 9},
    ]