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
