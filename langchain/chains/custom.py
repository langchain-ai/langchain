"""Custom chain class."""
from typing import Callable, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain


class SimpleCustomChain(Chain, BaseModel):
    """Custom chain with single string input/output."""

    func: Callable[[str], str]
    """Custom callable function."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        _input = inputs[self.input_key]
        output = self.func(_input)
        return {self.output_key: output}
