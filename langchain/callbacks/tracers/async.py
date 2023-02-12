"""Async callback tracers."""
import asyncio
from typing import Any, Union, Awaitable, Coroutine, List
import functools


class FooCallbackHandler:
    """FooCallbackHandler."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        print("on_llm_new_token", token, kwargs)


class FooCallbackHandler2(FooCallbackHandler):
    """FooCallbackHandler2."""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await asyncio.sleep(1)
        print("async on_llm_new_token", token, kwargs)


class FooCallbackManager(FooCallbackHandler):
    """FooCallbackManager."""

    def __init__(self, handlers: List[FooCallbackHandler]) -> None:
        """Initialize callback manager."""
        self.handlers: List[FooCallbackHandler] = handlers

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        for handler in self.handlers:
            if asyncio.iscoroutinefunction(handler.on_llm_new_token):
                asyncio.run(handler.on_llm_new_token(token, **kwargs))
            else:
                handler.on_llm_new_token(token, **kwargs)


class AsyncFooCallbackManager(FooCallbackHandler):
    """AsyncFooCallbackManager."""

    def __init__(self, handlers: List[FooCallbackHandler]) -> None:
        """Initialize callback manager."""
        self.handlers: List[FooCallbackHandler] = handlers

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        for handler in self.handlers:
            if asyncio.iscoroutinefunction(handler.on_llm_new_token):
                await handler.on_llm_new_token(token, **kwargs)
            else:
                # Run in executor
                await asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(handler.on_llm_new_token, token, **kwargs)
                )


def main() -> None:
    """Main."""
    print("FooCallbackManager")
    manager = FooCallbackManager([FooCallbackHandler(), FooCallbackHandler2()])
    manager.on_llm_new_token("foo", bar="baz")

    print("AsyncFooCallbackManager")
    manager = AsyncFooCallbackManager([FooCallbackHandler(), FooCallbackHandler2()])
    asyncio.run(manager.on_llm_new_token("foo", bar="baz"))


if __name__ == "__main__":
    main()
