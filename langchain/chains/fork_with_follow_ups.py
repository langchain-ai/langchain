"""Conditionally executes a follow up chain based on the output of a decision chain."""
from typing import Dict, List

from pydantic import BaseModel, Extra, validator

from langchain.chains import LLMChain
from langchain.chains.base import Chain


class ForkChain(Chain, BaseModel):
    """Conditionally executes follow up chain based on output of a decision chain."""

    decision_chain: LLMChain
    follow_up_chains: Dict[str, Chain]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator("follow_up_chains")
    def default_in_follow_up_chains(cls, v: Dict[str, Chain]) -> Dict[str, Chain]:
        """Make sure that `default` key exists in follow_up_chains."""
        if "default" not in v:
            raise ValueError(
                "`follow_up_chains` must contain a 'default' option. "
                "This is the chain that is called when the output of the "
                "decision chain doesn't match any key in `follow_up_chains`."
            )
        return v

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.decision_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return []

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        decision_chain_output = self.decision_chain.run(**inputs)
        try:
            return self.follow_up_chains[decision_chain_output.strip()](inputs)
        except KeyError:
            return self.follow_up_chains["default"](inputs)
