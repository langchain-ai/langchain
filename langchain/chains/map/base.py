"""Chain that generates a list and then maps each output to another chain."""

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from pydantic import BaseModel, Extra, root_validator
from typing import List, Dict


class MapChain(Chain, BaseModel):
    """Chain that generates a list and then maps each output to another chain."""

    llm_chain: LLMChain
    map_chain: Chain
    n: int
    output_key_prefix: str = "output"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        vars = self.llm_chain.prompt.input_variables
        return [v for v in vars if v != "n"]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [f"{self.output_key_prefix}_{i}" for i in range(self.n)]

    @root_validator()
    def validate_llm_chain(cls, values: Dict) -> Dict:
        """Check that llm chain takes as input `n`."""
        input_vars = values["llm_chain"].prompt.input_variables
        if "n" not in input_vars:
            raise ValueError(
                "For MapChains, `n` should be one of the input variables to "
                f"llm_chain, only got {input_vars}"
            )
        return values

    @root_validator()
    def validate_map_chain(cls, values: Dict) -> Dict:
        """Check that map chain takes a single input."""
        map_chain_inputs = values["map_chain"].input_keys
        if len(map_chain_inputs) != 1:
            raise ValueError(
                "For MapChains, the map_chain should take a single input,"
                f" got {map_chain_inputs}."
            )
        return values

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        _inputs = {key: inputs[key] for key in self.input_keys}
        _inputs["n"] = self.n
        output = self.llm_chain.predict(**_inputs)
        new_inputs = output.split("\n")
        if len(new_inputs) != self.n:
            raise ValueError(
                f"Got {len(new_inputs)} items, but expected to get {self.n}"
            )
        outputs = {self.map_chain.run(text) for text in new_inputs}
        return outputs

