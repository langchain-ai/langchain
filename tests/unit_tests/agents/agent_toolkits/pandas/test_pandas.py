import pandas as pd
import pytest

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.pandas.base import (
    _get_callbacks,
    create_pandas_dataframe_agent,
)
from langchain.callbacks.manager import CallbackManager
from tests.unit_tests.llms.fake_llm import FakeLLM


# Test cases for _get_callbacks function
def test_get_callbacks_default() -> None:
    callbacks = _get_callbacks(None, None, False)
    assert isinstance(callbacks, CallbackManager)


def test_get_callbacks_with_callback_manager() -> None:
    cb = CallbackManager.configure()
    with pytest.raises(ValueError):
        _get_callbacks(cb, cb, False)


def test_get_callbacks_deprecation_warning() -> None:
    cb = CallbackManager.configure()
    with pytest.warns(DeprecationWarning):
        _get_callbacks(None, cb, False)


# Test cases for create_pandas_dataframe_agent function
def test_create_agent_with_non_dataframe() -> None:
    with pytest.raises(ValueError, match="Expected pandas object, got"):
        create_pandas_dataframe_agent(FakeLLM(), "Not a dataframe")


def test_create_agent_with_dataframe() -> None:
    agent = create_pandas_dataframe_agent(FakeLLM(), pd.DataFrame())
    assert isinstance(agent, AgentExecutor)


def test_create_agent_with_callback_manager() -> None:
    cb = CallbackManager.configure()
    with pytest.raises(ValueError):
        create_pandas_dataframe_agent(
            FakeLLM(),
            pd.DataFrame(),
            callback_manager=cb,
            callbacks=cb,
        )
