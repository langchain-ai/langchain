from typing import Dict, List

from langchain.chains.pipeline import Pipeline
from langchain.chains.base import Chain
from pydantic import BaseModel


class FakeChain(Chain, BaseModel):

    input_variables: List[str]
    output_variables: List[str]

    @property
    def input_keys(self) -> List[str]:
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        return self.output_variables

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        outputs = {}
        for var in self.output_variables:
            outputs[var] = " ".join(self.input_variables) + "foo"
        return outputs


def test_pipeline_usage() -> None:
    chain_1 = FakeChain(input_variables=["foo"], output_variables=["bar"])
    chain_2 = FakeChain(input_variables=["bar"], output_variables=["baz"])
    pipeline = Pipeline(chains=[chain_1, chain_2], input_variables=["foo"])
    output = pipeline({"foo": "123"})
    breakpoint()