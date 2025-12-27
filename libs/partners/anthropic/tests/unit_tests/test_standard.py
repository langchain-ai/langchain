"""Standard LangChain interface tests."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]

from langchain_anthropic import ChatAnthropic

_MODEL = "claude-3-haiku-20240307"


class TestAnthropicStandard(ChatModelUnitTests):
    """Use the standard chat model unit tests against the `ChatAnthropic` class."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatAnthropic

    @property
    def chat_model_params(self) -> dict:
        return {"model": _MODEL}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"ANTHROPIC_API_KEY": "test"},
            {"model": _MODEL},
            {"anthropic_api_key": "test"},
        )


@pytest.mark.benchmark
def test_init_time_with_client(benchmark: BenchmarkFixture) -> None:
    """Test initialization time, accounting for lazy loading of client."""

    def _init_in_loop_with_clients() -> None:
        for _ in range(10):
            llm = ChatAnthropic(model="claude-3-5-haiku-20241022")
            _ = llm._client
            _ = llm._async_client

    benchmark(_init_in_loop_with_clients)
