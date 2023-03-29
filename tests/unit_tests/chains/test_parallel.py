"""Test parallel functionality."""
import random
import time
from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.parallel import ParallelChain


class FakeChain(Chain, BaseModel):
    """Fake Chain for testing purposes."""

    input_variables: List[str]
    output_variables: List[str]
    chain_id: int

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Input keys this chain returns."""
        return self.output_variables

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        time.sleep(random.uniform(1, 2))
        outputs = {}
        for var in self.output_variables:
            variables = [inputs[k] for k in self.input_variables]
            outputs[var] = f"{' '.join(variables)} {self.chain_id}"
        return outputs


def test_parallel_usage_single_input() -> None:
    """Test parallel on single input."""
    input_variables = ["input"]
    chain = ParallelChain(
        input_variables=input_variables,
        chains={
            "output1": FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out1"],
                chain_id=1,
            ),
            "output2": FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out2"],
                chain_id=2,
            ),
        },
    )
    output = chain("bar")
    expected_output = {
        "input": "bar",
        "output1/chain_out1": "bar 1",
        "output2/chain_out2": "bar 2",
    }
    assert output == expected_output


def test_parallel_usage_multiple_inputs() -> None:
    """Test parallel on multiple inputs."""
    input_variables = ["input1", "input2"]
    chain = ParallelChain(
        input_variables=input_variables,
        chains={
            "output1": FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out1"],
                chain_id=1,
            ),
            "output2": FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out2"],
                chain_id=2,
            ),
        },
    )
    inputs = {"input1": "foo", "input2": "bar"}
    output = chain(inputs)
    expected_output = {
        "output1/chain_out1": "foo bar 1",
        "output2/chain_out2": "foo bar 2",
    }
    assert output == {**inputs, **expected_output}


def test_parallel_usage_one_chain_single_output() -> None:
    """Test parallel on multiple inputs."""
    input_variables = ["input1", "input2"]
    chain = ParallelChain(
        input_variables=input_variables,
        chains={
            "output1": FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out1"],
                chain_id=1,
            ),
        },
    )
    inputs = {"input1": "foo", "input2": "bar"}
    output = chain(inputs)
    expected_output = {"output1/chain_out1": "foo bar 1"}
    assert output == {**inputs, **expected_output}


def test_parallel_error_zero_chains() -> None:
    """Test error is raised when there are no chains."""
    with pytest.raises(ValueError):
        ParallelChain(
            input_variables=["input1", "input2"],
            chains={},
        )


# async tests
def test_parallel_concurrency_speedup() -> None:
    """Test concurrent execution runs faster than serial execution."""
    num_child_chains = 10

    input_variables = ["input1", "input2"]

    chain = ParallelChain(
        input_variables=input_variables,
        chains={
            f"output{i}": FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out{i}"],
                chain_id=i,
            )
            for i in range(num_child_chains)
        },
        concurrent=True,
    )

    inputs = {"input1": "foo", "input2": "bar"}

    # measure time with concurrency
    start_time_concurrent = time.time()
    concurrent_output = chain(inputs)
    end_time_concurrent = time.time()

    # measure time without concurrency
    chain.concurrent = False
    start_time_serial = time.time()
    serial_output = chain(inputs)
    end_time_serial = time.time()

    # check that concurrent execution is faster.
    # Serial execution will run for >= 10 sec because each child chain sleeps >= sec.
    # Parallel execution will run for <= 2 sec because each child chain sleeps <= 2 sec.
    assert (
        end_time_concurrent - start_time_concurrent
        < end_time_serial - start_time_serial
    )
    assert concurrent_output == serial_output


def test_parallel_nested_speedup() -> None:
    """Test nested concurrent ParallelChains."""
    num_child_chains = 3

    input_variables = ["input1", "input2"]

    chain_concurrent = ParallelChain(
        input_variables=input_variables,
        chains={
            f"output{i}": ParallelChain(
                input_variables=input_variables,
                chains={
                    f"output{i}_{j}": FakeChain(
                        input_variables=input_variables,
                        output_variables=[f"chain_out{i}_{j}"],
                        chain_id=i * num_child_chains + j,
                    )
                    for j in range(num_child_chains)
                },
                concurrent=True,
            )
            for i in range(num_child_chains)
        },
        concurrent=True,
    )

    chain_serial = ParallelChain(
        input_variables=input_variables,
        chains={
            f"output{i}": ParallelChain(
                input_variables=input_variables,
                chains={
                    f"output{i}_{j}": FakeChain(
                        input_variables=input_variables,
                        output_variables=[f"chain_out{i}_{j}"],
                        chain_id=i * num_child_chains + j,
                    )
                    for j in range(num_child_chains)
                },
                concurrent=False,
            )
            for i in range(num_child_chains)
        },
        concurrent=False,
    )

    inputs = {"input1": "foo", "input2": "bar"}

    # measure time with concurrency
    start_time_concurrent = time.time()
    output_concurrent = chain_concurrent(inputs)
    end_time_concurrent = time.time()

    # measure time without concurrency
    start_time_serial = time.time()
    output_serial = chain_serial(inputs)
    end_time_serial = time.time()

    expected_output = {
        "input1": "foo",
        "input2": "bar",
        "output0/output0_0/chain_out0_0": "foo bar 0",
        "output0/output0_1/chain_out0_1": "foo bar 1",
        "output0/output0_2/chain_out0_2": "foo bar 2",
        "output1/output1_0/chain_out1_0": "foo bar 3",
        "output1/output1_1/chain_out1_1": "foo bar 4",
        "output1/output1_2/chain_out1_2": "foo bar 5",
        "output2/output2_0/chain_out2_0": "foo bar 6",
        "output2/output2_1/chain_out2_1": "foo bar 7",
        "output2/output2_2/chain_out2_2": "foo bar 8",
    }

    # check that concurrent execution is faster.
    # Serial execution will run for >= 10 sec because each child chain sleeps >= sec.
    # Parallel execution will run for <= 2 sec because each child chain sleeps <= 2 sec.
    assert (
        end_time_concurrent - start_time_concurrent
        < end_time_serial - start_time_serial
    )
    assert output_concurrent == output_serial == expected_output


# if we allow child chains to have different inputs, we should remove the following tests


def test_parallel_error_mismatched_inputs() -> None:
    """Test error is raised when input variables to the parallel chain do not match those of the child chains."""
    with pytest.raises(ValueError):
        ParallelChain(
            input_variables=["input1", "input3"],
            chains={
                "output1": FakeChain(
                    input_variables=["input1", "input2"],
                    output_variables=["chain_out1"],
                    chain_id=1,
                ),
                "output2": FakeChain(
                    input_variables=["input1", "input2"],
                    output_variables=["chain_out2"],
                    chain_id=2,
                ),
            },
        )


def test_parallel_error_mismatched_inputs_between_chains() -> None:
    """Test error is raised when the input variables to different chains do not match."""
    with pytest.raises(ValueError):
        ParallelChain(
            input_variables=["input1", "input2"],
            chains={
                "output1": FakeChain(
                    input_variables=["input1", "input2"],
                    output_variables=["chain_out1"],
                    chain_id=1,
                ),
                "output2": FakeChain(
                    input_variables=["input1", "input3"],
                    output_variables=["chain_out2"],
                    chain_id=2,
                ),
            },
        )
