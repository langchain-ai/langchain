"""Base callback handler that can be used to handle callbacks from langchain."""
import asyncio
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from langchain.schema import AgentAction, AgentFinish, LLMResult


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to handle callbacks from langchain."""

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return False

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @abstractmethod
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    @abstractmethod
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    @abstractmethod
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    @abstractmethod
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    @abstractmethod
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    @abstractmethod
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    @abstractmethod
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    @abstractmethod
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    @abstractmethod
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    @abstractmethod
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    @abstractmethod
    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    @abstractmethod
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    @abstractmethod
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""


class BaseCallbackManager(BaseCallbackHandler, ABC):
    """Base callback manager that can be used to handle callbacks from LangChain."""

    @property
    def is_async(self) -> bool:
        """Whether the callback manager is async."""
        return False

    @abstractmethod
    def add_handler(self, callback: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""

    @abstractmethod
    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""

    def set_handler(self, handler: BaseCallbackHandler) -> None:
        """Set handler as the only handler on the callback manager."""
        self.set_handlers([handler])

    @abstractmethod
    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""


class CallbackManager(BaseCallbackManager):
    """Callback manager that can be used to handle callbacks from langchain."""

    def __init__(self, handlers: List[BaseCallbackHandler]) -> None:
        """Initialize callback manager."""
        self.handlers: List[BaseCallbackHandler] = handlers

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    handler.on_llm_start(serialized, prompts, **kwargs)

    def on_llm_new_token(
        self, token: str, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when LLM generates a new token."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    handler.on_llm_new_token(token, **kwargs)

    def on_llm_end(
        self, response: LLMResult, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when LLM ends running."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    handler.on_llm_end(response)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    handler.on_llm_error(error)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        for handler in self.handlers:
            if not handler.ignore_chain:
                if verbose or handler.always_verbose:
                    handler.on_chain_start(serialized, inputs, **kwargs)

    def on_chain_end(
        self, outputs: Dict[str, Any], verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when chain ends running."""
        for handler in self.handlers:
            if not handler.ignore_chain:
                if verbose or handler.always_verbose:
                    handler.on_chain_end(outputs)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        for handler in self.handlers:
            if not handler.ignore_chain:
                if verbose or handler.always_verbose:
                    handler.on_chain_error(error)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    handler.on_tool_start(serialized, input_str, **kwargs)

    def on_agent_action(
        self, action: AgentAction, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    handler.on_agent_action(action, **kwargs)

    def on_tool_end(self, output: str, verbose: bool = False, **kwargs: Any) -> None:
        """Run when tool ends running."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    handler.on_tool_end(output, **kwargs)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    handler.on_tool_error(error)

    def on_text(self, text: str, verbose: bool = False, **kwargs: Any) -> None:
        """Run on additional input from chains and agents."""
        for handler in self.handlers:
            if verbose or handler.always_verbose:
                handler.on_text(text, **kwargs)

    def on_agent_finish(
        self, finish: AgentFinish, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    handler.on_agent_finish(finish, **kwargs)

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = handlers


class AsyncCallbackHandler(BaseCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    async def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    async def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    async def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    async def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Run on agent action."""

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""


class AsyncCallbackManager(BaseCallbackManager):
    """Async callback manager that can be used to handle callbacks from LangChain."""

    @property
    def is_async(self) -> bool:
        """Return whether the handler is async."""
        return True

    def __init__(self, handlers: List[BaseCallbackHandler]) -> None:
        """Initialize callback manager."""
        self.handlers: List[BaseCallbackHandler] = handlers

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_llm_start):
                        await handler.on_llm_start(serialized, prompts, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(
                                handler.on_llm_start, serialized, prompts, **kwargs
                            ),
                        )

    async def on_llm_new_token(
        self, token: str, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_llm_new_token):
                        await handler.on_llm_new_token(token, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(
                                handler.on_llm_new_token, token, **kwargs
                            ),
                        )

    async def on_llm_end(
        self, response: LLMResult, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when LLM ends running."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_llm_end):
                        await handler.on_llm_end(response, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(handler.on_llm_end, response, **kwargs),
                        )

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        for handler in self.handlers:
            if not handler.ignore_llm:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_llm_error):
                        await handler.on_llm_error(error, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(handler.on_llm_error, error, **kwargs),
                        )

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        for handler in self.handlers:
            if not handler.ignore_chain:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_chain_start):
                        await handler.on_chain_start(serialized, inputs, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(
                                handler.on_chain_start, serialized, inputs, **kwargs
                            ),
                        )

    async def on_chain_end(
        self, outputs: Dict[str, Any], verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when chain ends running."""
        for handler in self.handlers:
            if not handler.ignore_chain:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_chain_end):
                        await handler.on_chain_end(outputs, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(handler.on_chain_end, outputs, **kwargs),
                        )

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        for handler in self.handlers:
            if not handler.ignore_chain:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_chain_error):
                        await handler.on_chain_error(error, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(handler.on_chain_error, error, **kwargs),
                        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_tool_start):
                        await handler.on_tool_start(serialized, input_str, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(
                                handler.on_tool_start, serialized, input_str, **kwargs
                            ),
                        )

    async def on_tool_end(
        self, output: str, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when tool ends running."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_tool_end):
                        await handler.on_tool_end(output, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(handler.on_tool_end, output, **kwargs),
                        )

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_tool_error):
                        await handler.on_tool_error(error, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(handler.on_tool_error, error, **kwargs),
                        )

    async def on_text(self, text: str, verbose: bool = False, **kwargs: Any) -> None:
        """Run when text is printed."""
        for handler in self.handlers:
            if verbose or handler.always_verbose:
                if asyncio.iscoroutinefunction(handler.on_text):
                    await handler.on_text(text, **kwargs)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, functools.partial(handler.on_text, text, **kwargs)
                    )

    async def on_agent_action(
        self, action: AgentAction, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run on agent action."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_agent_action):
                        await handler.on_agent_action(action, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(
                                handler.on_agent_action, action, **kwargs
                            ),
                        )

    async def on_agent_finish(
        self, finish: AgentFinish, verbose: bool = False, **kwargs: Any
    ) -> None:
        """Run when agent finishes."""
        for handler in self.handlers:
            if not handler.ignore_agent:
                if verbose or handler.always_verbose:
                    if asyncio.iscoroutinefunction(handler.on_agent_finish):
                        await handler.on_agent_finish(finish, **kwargs)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            functools.partial(
                                handler.on_agent_finish, finish, **kwargs
                            ),
                        )

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = handlers
