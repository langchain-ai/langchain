"""Chain that runs an arbitrary python function."""
from typing import Any, Awaitable, Callable, Dict, List, Optional
import warnings

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
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
    """The keys expected by the transform's input dictionary."""
    output_variables: List[str]
    """The keys returned by the transform's output dictionary."""
    transform: Callable[[Dict[str, str]], Dict[str, str]]
    """The transform function."""
    coroutine: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None
    """The coroutine transform function."""

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

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if self.coroutine is not None:
            return await self.coroutine(inputs)
        else:
            warnings.warn(
                "TransformChain coroutine is None, falling back to synchronous transform"
            )
            return self.transform(inputs)
