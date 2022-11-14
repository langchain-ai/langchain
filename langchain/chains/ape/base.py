from typing import Dict, List

from langchain.chains.base import Chain
from pydantic import BaseModel, Extra
from langchain.llms.base import LLM
from langchain.chains.llm import LLMChain
from langchain.chains.ape.prompt import PROMPT


class APEChain(Chain, BaseModel):

    llm: LLM

    input_key: str = "code"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        chain = LLMChain(llm=self.llm, prompt=PROMPT)
        output = chain.run(inputs[self.input_key])
        return {self.output_key: output}

    def ape(self, examples: List[str]) -> str:
        combined_examples = "\n\n".join(examples)
        return self.run(combined_examples)

