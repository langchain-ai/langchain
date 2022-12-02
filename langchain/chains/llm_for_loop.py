"""Chain that first uses an LLM to generate multiple items then loops over them."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import ListOutputParser


class LLMForLoopChain(Chain, BaseModel):
    """Chain that first uses an LLM to generate multiple items then loops over them."""

    llm_chain: LLMChain
    """LLM chain to use to generate multiple items."""
    apply_chain: Chain
    """Chain to apply to each item that is generated."""
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.llm_chain.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    @root_validator()
    def validate_output_parser(cls, values: Dict) -> Dict:
        """Validate that the correct inputs exist for all chains."""
        chain = values["llm_chain"]
        if not isinstance(chain.prompt.output_parser, ListOutputParser):
            raise ValueError(
                f"The OutputParser on the base prompt should be of type "
                f"ListOutputParser, got {type(chain.prompt.output_parser)}"
            )
        return values

    def run_list(self, **kwargs: Any) -> List[str]:
        """Get list from LLM chain and then run chain on each item."""
        output_items = self.llm_chain.predict_and_parse(**kwargs)
        res = []
        for item in output_items:
            res.append(self.apply_chain.run(item))
        return res

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        res = self.run_list(**inputs)
        return {self.output_key: "\n\n".join(res)}
