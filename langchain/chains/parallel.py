import asyncio
import functools
import time
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain


class ParallelChain(Chain, BaseModel):
    """
    Chain topology where independent chains process inputs to produce multiple outputs.
    Each chain is run independently, possibly concurrently.
    Their outputs are merged together, with each output key of ParallelChain
    corresponding to a different chain's output.

    The word "parallel" is to be interpreted as "independent" rather than "concurrent",
    and refers to the topology of how the chains are connected (similar to how
    SequentialChain refers to the topology of how the chains are connected in sequence).

    Therefore, while the chains can be run concurrently, they are not run in parallel
    in the sense of being run on different threads or processes.
    """

    chains: Dict[str, Chain]
    concurrent: bool = True

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the union of all the input keys of the child chains.

        :meta private:
        """
        return list(set().union(*(chain.input_keys for chain in self.chains.values())))

    @property
    def output_keys(self) -> List[str]:
        """Return expected output keys of the chain.

        :meta private:
        """
        return list(self.chains.keys())

    @root_validator(pre=True)
    def validate_chains(cls, values: Dict) -> Dict:
        """Validate that there is at least one chain and all chains have the same input keys."""
        chains = values["chains"]

        if len(chains) == 0:
            raise ValueError("There must be at least one chain.")

        return values

    def _run_child(
        self, inputs: Dict[str, str], key: str, chain: Chain
    ) -> Dict[str, Any]:
        if self.verbose:
            print(f'Child chain for key="{key}" started.')
            t0 = time.time()
        # run chain only on the inputs that match the chain's input keys
        result = chain(
            {k: v for k, v in inputs.items() if k in chain.input_keys},
            return_only_outputs=True,
        )
        if self.verbose:
            print(
                f'Child chain for key="{key}" finished after {time.time() - t0:.2f} seconds.'
            )
        return {key: result}

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.concurrent:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as e:
                # to handle nested event loops
                if str(e).startswith("There is no current event loop in thread"):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                else:
                    raise e
            return loop.run_until_complete(self._acall(inputs))
        else:
            outputs = {}
            for key, chain in self.chains.items():
                outputs.update(self._run_child(inputs, key, chain))
            return outputs

    async def _arun_child(
        self,
        loop: asyncio.AbstractEventLoop,
        key: str,
        chain: Chain,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        func = functools.partial(self._run_child, key=key, chain=chain)
        result = await loop.run_in_executor(None, func, inputs)
        return result

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        tasks = []
        for key, chain in self.chains.items():
            tasks.append(loop.create_task(self._arun_child(loop, key, chain, inputs)))
        results = await asyncio.gather(*tasks)
        outputs = {}
        for result in results:
            outputs.update(result)
        return outputs
