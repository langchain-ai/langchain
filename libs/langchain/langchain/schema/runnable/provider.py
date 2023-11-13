from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain.load.dump import dumpd
from langchain.pydantic_v1 import BaseModel, PrivateAttr
from langchain.schema.runnable.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain.schema.runnable.config import (
    RunnableConfig,
    ensure_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain.schema.runnable.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)


class RunnableContextProvider(RunnableSerializable[Input, Output]):
    chain: Runnable[Input, Output]
    _started: bool = PrivateAttr(default=False)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    @property
    def started(self) -> bool:
        return self._started

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        self._started = True
        result = self.chain.invoke(input, config)
        self._started = False

        return result

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        self._started = True
        result = await self.chain.ainvoke(input, config, **kwargs)
        self._started = False

        return result

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        self._started = True
        result = self.chain.batch(inputs, config, return_exceptions=return_exceptions, **kwargs)
        self._started = False

        return result

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        self._started = True
        result = await self.chain.abatch(inputs, config, return_exceptions=return_exceptions, **kwargs)
        self._started = False

        return result


class RunnableContextBuilder(RunnableSerializable[Input, Output]):
    ...







