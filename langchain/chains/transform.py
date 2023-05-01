"""Chain that runs an arbitrary python function."""
from typing import Callable, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain


class TransformChain(Chain):
    """Chain transform chain output.

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
        """Expect input keys.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return output keys.

        :meta private:
        """
        return self.output_variables

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return self.transform(inputs)
