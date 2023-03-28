import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import time
import random
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator
from langchain.chains.base import Chain


class SimpleParallelChain(Chain, BaseModel):
    """
    Chain pipeline where multiple independent chains process the same inputs to produce multiple outputs.
    Each chain is run in parallel and their outputs are merged together,
    with each output key of SimpleParallelChain corresponding to a different chain's output.
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
        result = chain(inputs, return_only_outputs=True)
        return {f"{key}/{k}": v for k, v in result.items()}

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.concurrent:
            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(self._acall(inputs))
        else:
            outputs = {}
            for key, chain in self.chains.items():
                outputs.update(self._run_child(inputs, key, chain))
        return outputs

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        loop = asyncio.get_event_loop()
        tasks = []
        for key, chain in self.chains.items():
            tasks.append(loop.create_task(self._arun_child(loop, key, chain, inputs)))
        results = await asyncio.gather(*tasks)
        if self.verbose:
            print("All chains have finished.")
        outputs = {}
        for result in results:
            outputs.update(result)
        return outputs

    async def _arun_child(
        self,
        loop: asyncio.AbstractEventLoop,
        key: str,
        chain: Chain,
        inputs: Dict[str, str],
    ) -> Dict[str, str]:
        if self.verbose:
            print(f'Child chain for key="{key}" started.')
            t0 = time.time()
        # Run blocking function in a thread pool
        func = functools.partial(self._run_child, key=key, chain=chain)
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, func, inputs)
        if self.verbose:
            print(
                f'Child chain for key="{key}" finished after {time.time() - t0:.2f} seconds.'
            )
        return result
