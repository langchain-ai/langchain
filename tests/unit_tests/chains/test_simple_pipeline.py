"""Test functionality around the simple pipeline chain."""

from typing import Dict, List

import pytest
from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.sequential import SimplePipeline


class FakeChain(Chain, BaseModel):
    """Fake chain for testing purposes."""

    input_variables: List[str]
    output_variables: List[str]

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
            outputs[var] = " ".join(variables) + "foo"
        return outputs
