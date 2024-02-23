from abc import abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableSerializable,
)
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output


class RunnableFactory(RunnableSerializable[Input, Output]):
    """Runnable that creates another runnable at runtime.

    The runnable allows creating LCEL chains using object-oriented programming.
    It provides a factory method :meth:`_create_runnable` to build another runnable
    at runtime.

    Features:

    * Loadable parameters and validation thanks to ``pydantic``.
    * Inheritance support and all other class features.

    Example usage:

    .. code-block:: python

        from cogent_core.utils.runnable import RunnableFactory
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import Runnable
        from langchain_openai import ChatOpenAI
        from langchain_core.pydantic_v1 import validator

        class MyRunnable(RunnableFactory):
            role: str = "You are helpful AI assistant."
            prompt: str = "Tell me a joke about {topic}"
            model: str = "gpt-3.5-turbo"

            @validator("prompt")
            def validate_prompt(cls, v):
                if "joke" not in v:
                    raise ValueError("You should only ask for jokes.")
                return v

            def _create_runnable(self) -> Runnable:
                prompt = ChatPromptTemplate.from_messages(
                    [("system", self.role), ("human", self.prompt)]
                )
                llm = ChatOpenAI(model_name=self.model)

                chain = prompt | llm | StrOutputParser()
                return chain

        # Create a chain with defaults.

        chain = MyRunnable()
        output = chain.invoke({"topic": "bear"})
        print(output)

        # Create a chain with custom parameters.

        chain = MyRunnable(model="gpt-4-1106-preview")
        output = chain.invoke({"topic": "bear"})
        print(output)

        # Create a chain from external "source".

        external_parameters = {
            "model": "gpt-4-1106-preview",
            "prompt": "Tell me two jokes about {topic}",
        }

        chain = MyRunnable.parse_obj(external_parameters)
        output = chain.invoke({"topic": "bear"})
        print(output)
    """

    _runnable: Runnable = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._runnable = self._create_runnable()

    @abstractmethod
    def _create_runnable(self) -> Runnable:
        pass

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def InputType(self) -> type[Input]:
        return self._runnable.InputType

    @property
    def OutputType(self) -> type[Output]:
        return self._runnable.OutputType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        return self._runnable.get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        return self._runnable.get_output_schema(config)

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self._runnable.config_specs

    def config_schema(
        self, *, include: Optional[Sequence[str]] = None
    ) -> type[BaseModel]:
        return self._runnable.config_schema(include=include)

    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        return self._runnable.get_graph(config)

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        return self._runnable.transform(input, config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for chunk in self._runnable.atransform(input, config, **kwargs):
            yield chunk

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        return self._runnable.stream(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for chunk in self._runnable.astream(input, config, **kwargs):
            yield chunk

    def batch(
        self,
        inputs: list[Input],
        config: Optional[RunnableConfig | list[RunnableConfig]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        return self._runnable.batch(
            inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[RunnableConfig | list[RunnableConfig]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        return await self._runnable.abatch(
            inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return self._runnable.invoke(input, config)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        return await self._runnable.ainvoke(input, config, **kwargs)
