"""Chain pipeline where multiple independent chains process the same inputs to produce multiple outputs."""
from typing import Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.input import get_color_mapping


class SimpleParallelChain(Chain, BaseModel):
    """Chain pipeline where multiple independent chains process the same inputs to produce multiple outputs.

    This is a simple implementation of a parallel chain.
    It assumes each chain only produces one output.
    Each chain is run in parallel and their outputs are merged together,
    with each output key of SimpleParallelChain corresponding to a different chain's output.
    """
    chains: List[Chain]
    input_variables: List[str]  #: :meta private:
    output_variables: List[str]  #: :meta private:

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
        return self.output_variables

    @root_validator(pre=True)
    def validate_chains(cls, values: Dict) -> Dict:
        """Validate that
        - there is at least one chain
        - all chains take the same input
        - the number of output variables and number of chains are the same.
        """
        chains = values["chains"]

        if len(chains) == 0:
            raise ValueError("There must be at least one chain.")

        input_variables = values["input_variables"]
        for chain in chains:
            if chain.input_keys != input_variables:
                raise ValueError(
                    f"Chain {chain} has input keys {chain.input_keys} "
                    f"which do not match the expected input keys {input_variables}."
                )

            if len(chain.output_keys) != 1:
                raise ValueError(
                    f"Chain {chain} has {len(chain.output_keys)} output keys "
                    f"which is not supported by SimpleParallelChain."
                )

        if len(values["chains"]) != len(values["output_variables"]):
            raise ValueError(
                "The number of chains should exactly match the number of output variables."
                "There should be a 1-1 correspondence between each chain and output variable."
            )

        return values

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run each chain in parallel and merge their outputs together."""
        outputs = {}
        for key, chain in zip(self.output_variables, self.chains):
            chain_output = chain(inputs, return_only_outputs=True)
            only_output_key = next(iter(chain_output))  # we assume there is only one output key
            outputs[key] = chain_output[only_output_key]
        return outputs