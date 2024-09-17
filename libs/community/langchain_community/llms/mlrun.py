from collections.abc import AsyncIterator, Iterator
from typing import Any, List, Optional, Union

import mlrun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig


class Mlrun(LLM):
    """A class for interacting with Langchain, while using models served by mlrun.
    This class is used to interact with models served by mlrun, or just gated by mlrun.
    """

    llm: Any = Field(default=None)
    """The deployed mlrun model server object (nuclio function)."""
    name: str = Field(default=None)
    """The name to give to the model."""

    def __init__(
        self,
        nuclio_function: Any,
        name: str,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)
        self.llm = nuclio_function
        self.name = name

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model.
        Used for logging purposes only."""
        return "Mlrun"

    def _call(
        self,
        prompt: str,
        **generate_kwargs: dict,
    ) -> str:
        pass

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        call the llm invoke function, routed through mlrun model server.

        :param input:  The input to the model.
        :param config: The configuration to use while running the model.
        :param stop:   The stop words to use while running the model.
        """
        # Convert the input to the correct format
        input = self._convert_input(input)
        # Check if the model server is a mock server or a real server
        # and call the correct function
        if isinstance(self.llm, mlrun.serving.server.GraphServer):
            response = self.llm.test(
                path=f"/v2/models/{self.name}/predict",
                body={
                    "inputs": [input],
                    "stop": stop,
                    "config": config,
                    "usage": "invoke",
                    **kwargs,
                },
            )
            return response if type(response) is str else response["outputs"]
        else:
            response = self.llm.invoke(
                path=f"/v2/models/{self.name}/predict",
                body={
                    "inputs": [input],
                    "stop": stop,
                    "config": config,
                    "usage": "invoke",
                    **kwargs,
                },
            )
            return response if type(response) is str else response["outputs"]

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        call the llm ainvoke function, routed through mlrun model server.
        """
        raise NotImplementedError("ainvoke is not supported by mlrun yet.")

    def batch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        call the llm batch function, routed through mlrun model server.

        :param inputs:            The inputs to the model.
        :param config:            The configuration to use while running the model.
        :param return_exceptions: Whether to return exceptions or not.
        """
        # Convert all inputs to the correct format and list them
        inputs = [self._convert_input(input) for input in inputs]
        # Check if the model server is a mock server or a real server
        # and call the correct function
        if isinstance(self.llm, mlrun.serving.server.GraphServer):
            response = self.llm.test(
                path=f"/v2/models/{self.name}/predict",
                body={
                    "inputs": inputs,
                    "config": config,
                    "return_exceptions": return_exceptions,
                    "usage": "batch",
                    **kwargs,
                },
            )
            return response
        else:
            response = self.llm.invoke(
                path=f"/v2/models/{self.name}/predict",
                body={
                    "inputs": inputs,
                    "config": config,
                    "return_exceptions": return_exceptions,
                    "usage": "batch",
                    **kwargs,
                },
            )
            return response

    async def abatch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        call the llm abatch function, routed through mlrun model server.
        """
        raise NotImplementedError("abatch is not supported by mlrun yet.")

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        call the llm stream function, routed through mlrun model server.
        """
        raise NotImplementedError("Stream is not supported by mlrun yet.")

    def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        call the llm astream function, routed through mlrun model server.
        """
        raise NotImplementedError("Stream is not supported by mlrun yet.")
