"""Transform inputs to alternative outputs."""
from typing import Callable, Dict, List

from pydantic import BaseModel

from langchain.chains.base import Chain


class TransformChain(Chain, BaseModel):
    """Chain to transform chain output.

    Example:
        .. code-block:: python

            from langchain import TransformChain
            transform_chain = TransformChain(input_variables=["text"],
             output_variables["entities"], transform=func())
    """

    input_variables: List[str]
    output_variables: List[str]
    transform: Callable[[Dict[str, str]], Dict[str, str]]

    @property
    def input_keys(self) -> List[str]:
        """Expected input keys.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Returned output keys.

        :meta private:
        """
        return self.output_variables

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        missing_vars = set(inputs.keys()).difference(set(self.input_keys))
        if missing_vars:
            raise ValueError(f"Missing required input keys: {missing_vars}")

        return self.transform(inputs)
