"""Test CallbackManager."""
from typing import Tuple

import pytest

from langchain.callbacks.base import (
    AsyncCallbackManager,
    BaseCallbackManager,
    CallbackManager,
)
from langchain.callbacks.shared import SharedCallbackManager
from langchain.schema import AgentFinish, LLMResult
from tests.unit_tests.callbacks.fake_callback_handler import (
    BaseFakeCallbackHandler,
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


def _test_callback_manager(
    manager: BaseCallbackManager, *handlers: BaseFakeCallbackHandler
) -> None:
    """Test the CallbackManager."""
    manager.on_llm_start({}, [])
    manager.on_llm_end(LLMResult(generations=[]))
    manager.on_llm_error(Exception())
    manager.on_chain_start({"name": "foo"}, {})
    manager.on_chain_end({})
    manager.on_chain_error(Exception())
    manager.on_tool_start({}, "")
    manager.on_tool_end("")
    manager.on_tool_error(Exception())
    manager.on_agent_finish(AgentFinish(log="", return_values={}))
    _check_num_calls(handlers)


async def _test_callback_manager_async(
    manager: AsyncCallbackManager, *handlers: BaseFakeCallbackHandler
) -> None:
    """Test the CallbackManager."""
    await manager.on_llm_start({}, [])
    await manager.on_llm_end(LLMResult(generations=[]))
    await manager.on_llm_error(Exception())
    await manager.on_chain_start({"name": "foo"}, {})
    await manager.on_chain_end({})
    await manager.on_chain_error(Exception())
    await manager.on_tool_start({}, "")
    await manager.on_tool_end("")
    await manager.on_tool_error(Exception())
    await manager.on_agent_finish(AgentFinish(log="", return_values={}))
    _check_num_calls(handlers)


def _check_num_calls(handlers: Tuple[BaseFakeCallbackHandler, ...]) -> None:
    for handler in handlers:
        if handler.always_verbose:
            assert handler.starts == 3
            assert handler.ends == 4
            assert handler.errors == 3
        else:
            assert handler.starts == 0
            assert handler.ends == 0
            assert handler.errors == 0


def _test_callback_manager_pass_in_verbose(
    manager: BaseCallbackManager, *handlers: FakeCallbackHandler
) -> None:
    """Test the CallbackManager."""
    manager.on_llm_start({}, [], verbose=True)
    manager.on_llm_end(LLMResult(generations=[]), verbose=True)
    manager.on_llm_error(Exception(), verbose=True)
    manager.on_chain_start({"name": "foo"}, {}, verbose=True)
    manager.on_chain_end({}, verbose=True)
    manager.on_chain_error(Exception(), verbose=True)
    manager.on_tool_start({}, "", verbose=True)
    manager.on_tool_end("", verbose=True)
    manager.on_tool_error(Exception(), verbose=True)
    manager.on_agent_finish(AgentFinish(log="", return_values={}), verbose=True)
    for handler in handlers:
        assert handler.starts == 3
        assert handler.ends == 4
        assert handler.errors == 3


def test_callback_manager() -> None:
    """Test the CallbackManager."""
    handler1 = FakeCallbackHandler(always_verbose_=True)
    handler2 = FakeCallbackHandler(always_verbose_=False)
    manager = CallbackManager([handler1, handler2])
    _test_callback_manager(manager, handler1, handler2)


def test_callback_manager_pass_in_verbose() -> None:
    """Test the CallbackManager."""
    handler1 = FakeCallbackHandler()
    handler2 = FakeCallbackHandler()
    manager = CallbackManager([handler1, handler2])
    _test_callback_manager_pass_in_verbose(manager, handler1, handler2)


def test_ignore_llm() -> None:
    """Test ignore llm param for callback handlers."""
    handler1 = FakeCallbackHandler(ignore_llm_=True, always_verbose_=True)
    handler2 = FakeCallbackHandler(always_verbose_=True)
    manager = CallbackManager(handlers=[handler1, handler2])
    manager.on_llm_start({}, [], verbose=True)
    manager.on_llm_end(LLMResult(generations=[]), verbose=True)
    manager.on_llm_error(Exception(), verbose=True)
    assert handler1.starts == 0
    assert handler1.ends == 0
    assert handler1.errors == 0
    assert handler2.starts == 1
    assert handler2.ends == 1
    assert handler2.errors == 1


def test_ignore_chain() -> None:
    """Test ignore chain param for callback handlers."""
    handler1 = FakeCallbackHandler(ignore_chain_=True, always_verbose_=True)
    handler2 = FakeCallbackHandler(always_verbose_=True)
    manager = CallbackManager(handlers=[handler1, handler2])
    manager.on_chain_start({"name": "foo"}, {}, verbose=True)
    manager.on_chain_end({}, verbose=True)
    manager.on_chain_error(Exception(), verbose=True)
    assert handler1.starts == 0
    assert handler1.ends == 0
    assert handler1.errors == 0
    assert handler2.starts == 1
    assert handler2.ends == 1
    assert handler2.errors == 1


def test_ignore_agent() -> None:
    """Test ignore agent param for callback handlers."""
    handler1 = FakeCallbackHandler(ignore_agent_=True, always_verbose_=True)
    handler2 = FakeCallbackHandler(always_verbose_=True)
    manager = CallbackManager(handlers=[handler1, handler2])
    manager.on_tool_start({}, "", verbose=True)
    manager.on_tool_end("", verbose=True)
    manager.on_tool_error(Exception(), verbose=True)
    manager.on_agent_finish(AgentFinish({}, ""), verbose=True)
    assert handler1.starts == 0
    assert handler1.ends == 0
    assert handler1.errors == 0
    assert handler2.starts == 1
    assert handler2.ends == 2
    assert handler2.errors == 1


def test_shared_callback_manager() -> None:
    """Test the SharedCallbackManager."""
    manager1 = SharedCallbackManager()
    manager2 = SharedCallbackManager()

    assert manager1 is manager2

    handler1 = FakeCallbackHandler(always_verbose_=True)
    handler2 = FakeCallbackHandler()
    manager1.add_handler(handler1)
    manager2.add_handler(handler2)
    _test_callback_manager(manager1, handler1, handler2)


@pytest.mark.asyncio
async def test_async_callback_manager() -> None:
    """Test the AsyncCallbackManager."""
    handler1 = FakeAsyncCallbackHandler(always_verbose_=True)
    handler2 = FakeAsyncCallbackHandler()
    manager = AsyncCallbackManager([handler1, handler2])
    await _test_callback_manager_async(manager, handler1, handler2)


@pytest.mark.asyncio
async def test_async_callback_manager_sync_handler() -> None:
    """Test the AsyncCallbackManager."""
    handler1 = FakeCallbackHandler(always_verbose_=True)
    handler2 = FakeAsyncCallbackHandler()
    manager = AsyncCallbackManager([handler1, handler2])
    await _test_callback_manager_async(manager, handler1, handler2)
