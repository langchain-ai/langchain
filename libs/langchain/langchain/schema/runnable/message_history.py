import asyncio
from typing import Any, Callable, Dict, List, Optional, Type

from langchain.callbacks.tracers.schemas import Run
from langchain.pydantic_v1 import BaseModel, create_model
from langchain.schema.chat_history import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration, ChatResult, LLMResult
from langchain.schema.runnable.base import Runnable, RunnableBinding, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable.passthrough import RunnablePassthrough


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
        history_key: str = "history",
    ) -> None:
        bound = RunnablePassthrough.assign(
            **{history_key: RunnableLambda(self._enter_history, self._aenter_history)}
        ) | runnable.with_listeners(on_end=self._exit_history)
        super().__init__(
            factory=factory,
            input_key=input_key,
            output_key=output_key,
            bound=bound,
            kwargs={},
        )

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        super_schema = super().get_input_schema(config)
        if super_schema.__custom_root_type__ is not None:
            # The schema is not correct so we'll default to dict with input_key
            return create_model(  # type: ignore[call-overload]
                "RunnableWithChatHistoryInput",
                **{self.input_key: (str, ...)},
            )
        else:
            return super_schema

    def _enter_history(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> List[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        return hist.messages.copy()

    async def _aenter_history(
        self, input: Dict[str, Any], config: RunnableConfig
    ) -> List[BaseMessage]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._enter_history, input, config
        )

    def _exit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        # Add the input message
        hist.add_user_message(run.inputs[self.input_key])
        # Find the output message
        assert run.outputs is not None
        if self.output_key is None:
            assert len(run.outputs) == 1
            output = list(run.outputs.values())[0]
        else:
            output = run.outputs[self.output_key]
        # Add the output message
        if isinstance(output, str):
            hist.add_ai_message(output)
        elif isinstance(output, BaseMessage):
            hist.add_message(output)
        elif isinstance(output, ChatGeneration):
            hist.add_message(output.message)
        elif isinstance(output, ChatResult):
            hist.add_message(output.generations[0].message)
        elif isinstance(output, LLMResult):
            hist.add_ai_message(output.generations[0][0].text)
        else:
            raise ValueError(f"Unknown output type {type(output)}")

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        # extract session_id
        config["configurable"] = config.get("configurable", {})
        try:
            session_id = config["configurable"]["session_id"]
        except KeyError:
            example_input = {self.input_key: "foo"}
            raise ValueError(
                "session_id is required when using .with_message_history()"
                "\nPass it in as part of the config argument to .invoke() or .stream()"
                f'\neg. chain.invoke({example_input}, {{"configurable": {{"session_id":'
                ' "123"}})'
            )
        del config["configurable"]["session_id"]
        # attach message_history
        config["configurable"]["message_history"] = self.factory(  # type: ignore
            session_id=session_id,
        )
        return config
