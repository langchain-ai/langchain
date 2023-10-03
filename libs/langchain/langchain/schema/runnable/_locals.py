from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Union,
)

from langchain.schema.runnable.base import Input, Output, RunnableSerializable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.passthrough import RunnablePassthrough

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


class PutLocalVar(RunnablePassthrough):
    key: Union[str, Mapping[str, str]]
    """The key(s) to use for storing the input variable(s) in local state.
    
    If a string is provided then the entire input is stored under that key. If a 
        Mapping is provided, then the map values are gotten from the input and 
        stored in local state under the map keys.
    """

    def __init__(self, key: Union[str, Mapping[str, str]], **kwargs: Any) -> None:
        super().__init__(key=key, **kwargs)

    def _concat_put(
        self,
        input: Input,
        *,
        config: Optional[RunnableConfig] = None,
        replace: bool = False,
    ) -> None:
        if config is None:
            raise ValueError(
                "PutLocalVar should only be used in a RunnableSequence, and should "
                "therefore always receive a non-null config."
            )
        if isinstance(self.key, str):
            if self.key not in config["locals"] or replace:
                config["locals"][self.key] = input
            else:
                config["locals"][self.key] += input
        elif isinstance(self.key, Mapping):
            if not isinstance(input, Mapping):
                raise TypeError(
                    f"Received key of type Mapping but input of type {type(input)}. "
                    f"input is expected to be of type Mapping when key is Mapping."
                )
            for input_key, put_key in self.key.items():
                if put_key not in config["locals"] or replace:
                    config["locals"][put_key] = input[input_key]
                else:
                    config["locals"][put_key] += input[input_key]
        else:
            raise TypeError(
                f"`key` should be a string or Mapping[str, str], received type "
                f"{(type(self.key))}."
            )

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Input:
        self._concat_put(input, config=config, replace=True)
        return super().invoke(input, config=config)

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Input:
        self._concat_put(input, config=config, replace=True)
        return await super().ainvoke(input, config=config)

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Input]:
        for chunk in super().transform(input, config=config):
            self._concat_put(chunk, config=config)
            yield chunk

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Input]:
        async for chunk in super().atransform(input, config=config):
            self._concat_put(chunk, config=config)
            yield chunk


class GetLocalVar(
    RunnableSerializable[Input, Union[Output, Dict[str, Union[Input, Output]]]]
):
    key: str
    """The key to extract from the local state."""
    passthrough_key: Optional[str] = None
    """The key to use for passing through the invocation input. 
    
    If None, then only the value retrieved from local state is returned. Otherwise a 
        dictionary ``{self.key: <<retrieved_value>>, self.passthrough_key: <<input>>}``
        is returned.
    """

    def __init__(self, key: str, **kwargs: Any) -> None:
        super().__init__(key=key, **kwargs)

    def _get(
        self,
        input: Input,
        run_manager: Union[CallbackManagerForChainRun, Any],
        config: RunnableConfig,
    ) -> Union[Output, Dict[str, Union[Input, Output]]]:
        if self.passthrough_key:
            return {
                self.key: config["locals"][self.key],
                self.passthrough_key: input,
            }
        else:
            return config["locals"][self.key]

    async def _aget(
        self,
        input: Input,
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Union[Output, Dict[str, Union[Input, Output]]]:
        return self._get(input, run_manager, config)

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Union[Output, Dict[str, Union[Input, Output]]]:
        if config is None:
            raise ValueError(
                "GetLocalVar should only be used in a RunnableSequence, and should "
                "therefore always receive a non-null config."
            )

        return self._call_with_config(self._get, input, config)

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Union[Output, Dict[str, Union[Input, Output]]]:
        if config is None:
            raise ValueError(
                "GetLocalVar should only be used in a RunnableSequence, and should "
                "therefore always receive a non-null config."
            )

        return await self._acall_with_config(self._aget, input, config)
