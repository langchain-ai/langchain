import concurrent.futures
import random
import time
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain


class ParallelChain(Chain, BaseModel):
    """
    Chain pipeline where multiple independent chains process the same inputs to produce multiple outputs.
    Each chain is run in parallel and their outputs are merged together,
    with each output key of ParallelChain corresponding to a different chain's output.
    """

    input_variables: List[str]  #: :meta private:
    chains: Dict[str, Chain]
    concurrent: bool = True

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return expected input keys to each chain, which should all be the same.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return expected output keys of the chain.

        :meta private:
        """
        return [
            f"{key}/{k}"
            for key in self.chains.keys()
            for k in self.chains[key].output_keys
        ]

    @root_validator(pre=True)
    def validate_chains(cls, values: Dict) -> Dict:
        """Validate that there is at least one chain and all chains have the same input keys."""
        chains = values["chains"]

        if len(chains) == 0:
            raise ValueError("There must be at least one chain.")

        input_variables = values["input_variables"]
        for chain in chains.values():
            if chain.input_keys != input_variables:
                raise ValueError(
                    f"Chain {chain} has input keys {chain.input_keys} "
                    f"which do not match the expected input keys {input_variables}."
                )

        return values

    def _run_child(
        self, inputs: Dict[str, str], key: str, chain: Chain
    ) -> Dict[str, str]:
        if self.verbose:
            print(f'Child chain for key="{key}" started.')
            t0 = time.time()
        result = chain(inputs, return_only_outputs=True)
        if self.verbose:
            print(
                f'Child chain for key="{key}" finished after {time.time() - t0:.2f} seconds.'
            )
        return {f"{key}/{k}": v for k, v in result.items()}

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.concurrent:
            outputs = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.chains)
            ) as executor:
                futures = {
                    executor.submit(self._run_child, inputs, key, chain): key
                    for key, chain in self.chains.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    try:
                        outputs.update(future.result())
                    except Exception as exc:
                        print(f"Chain {key} generated an exception: {exc}")
        else:
            outputs = {}
            for key, chain in self.chains.items():
                outputs.update(self._run_child(inputs, key, chain))
        return outputs


if __name__ == "__main__":
    import pprint

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

    num_child_chains = 3

    input_variables = ["input1", "input2"]

    chain = ParallelChain(
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
                verbose=True,
                concurrent=False,
            )
            for i in range(num_child_chains)
        },
        verbose=True,
        concurrent=False,
    )

    inputs = {"input1": "foo", "input2": "bar"}

    # measure time with concurrency
    start_time_concurrent = time.time()
    output = chain(inputs)
    end_time_concurrent = time.time()

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

    pprint.pprint(output)
    pprint.pprint(expected_output)
    print(f"That took {end_time_concurrent - start_time_concurrent:.2f} seconds")
