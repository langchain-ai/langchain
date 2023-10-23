import asyncio
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

from langchain.schema.chat_history import BaseChatMessageHistory
from langchain.schema.runnable.base import Runnable, RunnableBinding, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig, merge_configs, patch_config
from langchain.schema.runnable.passthrough import RunnablePassthrough
from langchain.schema.runnable.utils import Input, Output


class RunnableWithMessageHistory(RunnableBinding):
    factory: Callable[[str], BaseChatMessageHistory]

    input_key: str

    output_key: Optional[str]

    def __init__(
        self,
        runnable: Runnable,
        factory: Callable[[str], BaseChatMessageHistory],
        input_key: str,
        output_key: Optional[str] = None,
    ) -> None:
        bound = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self._enter_history, self._aenter_history)
            )
            | runnable
            | RunnablePassthrough(self._exit_history, self._aexit_history)
        )
        super().__init__(
            factory=factory,
            input_key=input_key,
            output_key=output_key,
            bound=bound,
            kwargs={},
        )

    def _enter_history(self, input: Dict[str, Any], config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        messages = hist.messages

        hist.add_user_message(input[self.input_key])

        return messages

    def _exit_history(
        self, input: Union[str, Dict[str, Any]], config: RunnableConfig
    ) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        hist.add_ai_message(
            input if self.output_key is None else input[self.output_key]
        )

    async def _aenter_history(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> None:
        return asyncio.get_running_loop().run_in_executor(
            None, self._enter_history, input, config
        )

    async def _aexit_history(self, input: str, config: RunnableConfig) -> None:
        return asyncio.get_running_loop().run_in_executor(
            None, self._exit_history, input, config
        )

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = merge_configs(*configs)

        config["configurable"] = config.get("configurable", {})
        session_id = config["configurable"]["session_id"]

        del config["configurable"]["session_id"]
        config["configurable"]["message_history"] = self.factory(session_id=session_id)

        return config

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return self.bound.invoke(
            input,
            self._merge_configs(self.config, config),
            **{**self.kwargs, **kwargs},
        )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return await self.bound.ainvoke(
            input,
            self._merge_configs(self.config, config),
            **{**self.kwargs, **kwargs},
        )

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        if isinstance(config, list):
            configs = cast(
                List[RunnableConfig],
                [self._merge_configs(self.config, conf) for conf in config],
            )
        else:
            # TODO this case needs to raise exception
            configs = [
                patch_config(self._merge_configs(self.config, config), copy_locals=True)
                for _ in range(len(inputs))
            ]
        return self.bound.batch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        if isinstance(config, list):
            configs = cast(
                List[RunnableConfig],
                [self._merge_configs(self.config, conf) for conf in config],
            )
        else:
            # TODO this case needs to raise exception
            configs = [
                patch_config(self._merge_configs(self.config, config), copy_locals=True)
                for _ in range(len(inputs))
            ]
        return await self.bound.abatch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self.bound.stream(
            input,
            self._merge_configs(self.config, config),
            **{**self.kwargs, **kwargs},
        )

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for item in self.bound.astream(
            input,
            self._merge_configs(self.config, config),
            **{**self.kwargs, **kwargs},
        ):
            yield item

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        yield from self.bound.transform(
            input,
            self._merge_configs(self.config, config),
            **{**self.kwargs, **kwargs},
        )

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async for item in self.bound.atransform(
            input,
            self._merge_configs(self.config, config),
            **{**self.kwargs, **kwargs},
        ):
            yield item
