"""Simple chain pipeline where the outputs of one step feed directly into next."""

from langchain.chains.base import Chain
from pydantic import BaseModel, Extra, root_validator
from typing import List, Dict


class Pipeline(Chain, BaseModel):
    """Chain pipeline where the outputs of one step feed directly into next."""

    chains: List[Chain]
    input_variables: List[str]
    output_variables: List[str]  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return self.output_variables

    @root_validator(pre=True)
    def validate_chains(cls, values: Dict) -> Dict:
        """Validate that chains are all single input/output."""
        chains = values["chains"]
        input_variables = values["input_variables"]
        known_variables = set(input_variables)
        for chain in chains:
            missing_vars = set(chain.input_keys).difference(known_variables)
            if missing_vars:
                raise ValueError(f"Missing required input keys: {missing_vars}")
            overlapping_keys = known_variables.intersection(chain.output_keys)
            if overlapping_keys:
                raise ValueError(f"Chain returned keys that already exist: {overlapping_keys}")
            known_variables |= set(chain.output_keys)

        if "output_variables" not in values:
            values["output_variables"] = known_variables.difference(input_variables)
        else:
            missing_vars = known_variables.difference(values["output_variables"])
            if missing_vars:
                raise ValueError(f"Expected output variables that were not found: {missing_vars}.")
        return values

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        known_values = inputs.copy()
        for chain in self.chains:
            outputs = chain(known_values)
            known_values.update(outputs)
        return {k: known_values[k] for k in self.output_variables}


