"""Test parallel functionality."""
from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.parallel import SimpleParallelChain


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
        outputs = {}
        for var in self.output_variables:
            variables = [inputs[k] for k in self.input_variables]
            outputs[var] = f"{' '.join(variables)} {self.chain_id}"
        return outputs


def test_parallel_usage_single_input() -> None:
    """Test parallel on single input."""
    input_variables = ["input"]
    chain = SimpleParallelChain(
        chains=[
            FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out1"],
                chain_id=1,
            ),
            FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out2"],
                chain_id=2,
            ),
        ],
        input_variables=input_variables,
        output_variables=["output1", "output2"],
    )
    output = chain("bar")
    expected_output = {"input": "bar", "output1": "bar 1", "output2": "bar 2"}
    assert output == expected_output


def test_parallel_usage_multiple_inputs() -> None:
    """Test parallel on multiple inputs."""
    input_variables = ["input1", "input2"]
    chain = SimpleParallelChain(
        chains=[
            FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out1"],
                chain_id=1,
            ),
            FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out2"],
                chain_id=2,
            ),
        ],
        input_variables=input_variables,
        output_variables=["output1", "output2"],
    )
    inputs = {"input1": "foo", "input2": "bar"}
    output = chain(inputs)
    expected_output = {"output1": "foo bar 1", "output2": "foo bar 2"}
    assert output == {**inputs, **expected_output}


def test_parallel_usage_one_chain_single_output() -> None:
    """Test parallel on multiple inputs."""
    input_variables = ["input1", "input2"]
    chain = SimpleParallelChain(
        chains=[
            FakeChain(
                input_variables=input_variables,
                output_variables=["chain_out1"],
                chain_id=1,
            ),
        ],
        input_variables=input_variables,
        output_variables=["output1"],
    )
    inputs = {"input1": "foo", "input2": "bar"}
    output = chain(inputs)
    expected_output = {"output1": "foo bar 1"}
    assert output == {**inputs, **expected_output}


def test_parallel_error_mismatched_inputs() -> None:
    """Test error is raised when the input variables to the parallel chain do not match the input variables to the chains."""
    with pytest.raises(ValueError):
        SimpleParallelChain(
            chains=[
                FakeChain(
                    input_variables=["input1", "input2"],
                    output_variables=["chain_out1"],
                    chain_id=1,
                ),
                FakeChain(
                    input_variables=["input1", "input2"],
                    output_variables=["chain_out2"],
                    chain_id=2,
                ),
            ],
            input_variables=["input1", "input3"],
            output_variables=["output1", "output2"],
        )


def test_parallel_error_mismatched_inputs_between_chains() -> None:
    """Test error is raised when the input variables to different chains do not match."""
    with pytest.raises(ValueError):
        SimpleParallelChain(
            chains=[
                FakeChain(
                    input_variables=["input1", "input2"],
                    output_variables=["chain_out1"],
                    chain_id=1,
                ),
                FakeChain(
                    input_variables=["input1", "input3"],
                    output_variables=["chain_out2"],
                    chain_id=2,
                ),
            ],
            input_variables=["input1", "input2"],
            output_variables=["output1", "output2"],
        )


def test_parallel_error_single_chain_multiple_outputs() -> None:
    """Test error is raised when there is a single chain but only multiple outputs."""
    input_variables = ["input1", "input2"]
    with pytest.raises(ValueError):
        SimpleParallelChain(
            chains=[
                FakeChain(
                    input_variables=input_variables,
                    output_variables=["chain_out1"],
                    chain_id=1,
                ),
            ],
            input_variables=input_variables,
            output_variables=["output1", "output2"],
        )


def test_parallel_error_multiple_chains_single_output() -> None:
    """Test error is raised when there are multiple chains but only a single output."""
    input_variables = ["input1", "input2"]
    with pytest.raises(ValueError):
        SimpleParallelChain(
            chains=[
                FakeChain(
                    input_variables=input_variables,
                    output_variables=["chain_out1"],
                    chain_id=1,
                ),
                FakeChain(
                    input_variables=input_variables,
                    output_variables=["chain_out2"],
                    chain_id=2,
                ),
            ],
            input_variables=input_variables,
            output_variables=["output1"],
        )


def test_parallel_error_different_number_of_chains_than_outputs() -> None:
    """Test error is raised when there are different numbers of chains than outputs."""
    input_variables = ["input1", "input2"]
    with pytest.raises(ValueError):
        SimpleParallelChain(
            chains=[
                FakeChain(
                    input_variables=input_variables,
                    output_variables=["chain_out1"],
                    chain_id=1,
                ),
                FakeChain(
                    input_variables=input_variables,
                    output_variables=["chain_out2"],
                    chain_id=2,
                ),
            ],
            input_variables=input_variables,
            output_variables=["output1", "output2", "output3"],
        )


def test_parallel_error_zero_chains() -> None:
    """Test error is raised when there are no chains."""
    with pytest.raises(ValueError):
        SimpleParallelChain(
            chains=[],
            input_variables=["input1", "input2"],
            output_variables=["output1", "output2"],
        )
