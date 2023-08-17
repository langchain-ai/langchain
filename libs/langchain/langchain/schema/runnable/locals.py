from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional, Union

from langchain.load.serializable import Serializable
from langchain.schema.runnable.base import Input, Output, Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.passthrough import RunnablePassthrough


class PutLocalVar(RunnablePassthrough):
    key: Union[str, Mapping[str, str]]
    """The key(s) to use for storing the input variable(s) in local state.
    
    If a string is provided then the entire input is stored under that key. If a 
        Mapping is provided, then the map values are gotten from the input and 
        stored in local state under the map keys.
    """

    def __init__(self, key: Union[str, Mapping[str, str]], **kwargs: Any) -> None:
        super().__init__(key=key, **kwargs)

    def _put(self, input: Input, *, config: Optional[RunnableConfig] = None) -> None:
        if config is None:
            raise ValueError(
                "PutLocalVar should only be used in a RunnableSequence, and should "
                "therefore always receive a non-null config."
            )
        if isinstance(self.key, str):
            config["_locals"][self.key] = input
        elif isinstance(self.key, Mapping):
            if not isinstance(input, Mapping):
                raise TypeError(
                    f"Received key of type Mapping but input of type {type(input)}. "
                    f"input is expected to be of type Mapping when key is Mapping."
                )
            for input_key, put_key in self.key.items():
                config["_locals"][put_key] = input[input_key]
        else:
            raise TypeError(
                f"`key` should be a string or Mapping[str, str], received type "
                f"{(type(self.key))}."
            )

    def _concat_put(
        self, input: Input, *, config: Optional[RunnableConfig] = None
    ) -> None:
        if config is None:
            raise ValueError(
                "PutLocalVar should only be used in a RunnableSequence, and should "
                "therefore always receive a non-null config."
            )
        print(config)
        if isinstance(self.key, str):
            if self.key not in config["_locals"]:
                config["_locals"][self.key] = input
            else:
                config["_locals"][self.key] += input
        elif isinstance(self.key, Mapping):
            if not isinstance(input, Mapping):
                raise TypeError(
                    f"Received key of type Mapping but input of type {type(input)}. "
                    f"input is expected to be of type Mapping when key is Mapping."
                )
            for input_key, put_key in self.key.items():
                if put_key not in config["_locals"]:
                    config["_locals"][put_key] = input[input_key]
                else:
                    config["_locals"][put_key] += input[input_key]
        else:
            raise TypeError(
                f"`key` should be a string or Mapping[str, str], received type "
                f"{(type(self.key))}."
            )

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Input:
        self._put(input, config=config)
        return super().invoke(input, config=config)

    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None
    ) -> Input:
        self._put(input, config=config)
        return await super().ainvoke(input, config=config)

    def transform(
        self, input: Iterator[Input], config: RunnableConfig | None = None
    ) -> Iterator[Input]:
        for chunk in super().transform(input, config=config):
            self._concat_put(chunk, config=config)
            yield chunk

    async def atransform(
        self, input: AsyncIterator[Input], config: RunnableConfig | None = None
    ) -> AsyncIterator[Input]:
        async for chunk in super().atransform(input, config=config):
            self._concat_put(chunk, config=config)
            yield chunk


class GetLocalVar(
    Serializable, Runnable[Input, Union[Output, Dict[str, Union[Input, Output]]]]
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

    def _get(self, full_input: Dict) -> Union[Output, Dict[str, Union[Input, Output]]]:
        if self.passthrough_key:
            return {
                self.key: full_input["locals"][self.key],
                self.passthrough_key: full_input["input"],
            }
        else:
            return full_input["locals"][self.key]

    async def _aget(
        self, full_input: Dict
    ) -> Union[Output, Dict[str, Union[Input, Output]]]:
        return self._get(full_input)

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Union[Output, Dict[str, Union[Input, Output]]]:
        if config is None:
            raise ValueError(
                "GetLocalVar should only be used in a RunnableSequence, and should "
                "therefore always receive a non-null config."
            )

        log_input = {"input": input, "locals": config["_locals"]}
        return self._call_with_config(self._get, log_input, config)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Union[Output, Dict[str, Union[Input, Output]]]:
        if config is None:
            raise ValueError(
                "GetLocalVar should only be used in a RunnableSequence, and should "
                "therefore always receive a non-null config."
            )

        log_input = {"input": input, "locals": config["_locals"]}
        return await self._acall_with_config(self._aget, log_input, config)
