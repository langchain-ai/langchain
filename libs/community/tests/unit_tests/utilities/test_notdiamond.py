import random
import sys
import uuid
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models.llms import LLM

sys.modules["notdiamond"] = MagicMock()

from langchain_community.utilities.notdiamond import (  # noqa: E402
    NotDiamondRoutedRunnable,
    NotDiamondRunnable,
    _nd_provider_to_langchain_provider,
)


@pytest.fixture
def llm_configs() -> List[Any]:
    return [
        "openai/gpt-4o",
        "anthropic/claude-3-opus-20240229",
        "google/gemini-1.5-pro-latest",
    ]


@pytest.fixture
def nd_client(llm_configs: List[Any]) -> Any:
    client = MagicMock(llm_configs=llm_configs, api_key="", default="openai/gpt-4o")
    selected_model = random.choice(llm_configs)
    client.chat.completions.model_select = MagicMock(
        return_value=(uuid.uuid4(), selected_model)
    )
    client.chat.completions.amodel_select = AsyncMock(
        return_value=(uuid.uuid4(), selected_model)
    )
    return client


@pytest.fixture
def not_diamond_runnable(nd_client: Any) -> NotDiamondRunnable:
    with patch("langchain_community.utilities.notdiamond.LLMConfig") as mock_llm_config:
        mock_llm_config.from_string.return_value = MagicMock(provider="openai")
        runnable = NotDiamondRunnable(
            nd_client=nd_client, nd_kwargs={"tradeoff": "cost"}
        )
    return runnable


@pytest.fixture
def not_diamond_routed_runnable(nd_client: Any) -> NotDiamondRoutedRunnable:
    with patch("langchain_community.utilities.notdiamond.LLMConfig") as mock_llm_config:
        mock_llm_config.from_string.return_value = MagicMock(provider="openai")
        routed_runnable = NotDiamondRoutedRunnable(
            nd_client=nd_client, nd_kwargs={"tradeoff": "cost"}
        )
        routed_runnable._configurable_model = MagicMock(spec=_ConfigurableModel)
    return routed_runnable


class TestNotDiamondRunnable:
    def test_model_select(
        self,
        not_diamond_runnable: NotDiamondRunnable,
        llm_configs: List,
        nd_client: Any,
    ) -> None:
        prompt = "Hello, world!"
        actual_select = not_diamond_runnable._model_select(prompt)
        assert str(actual_select) in [
            _nd_provider_to_langchain_provider(str(config)) for config in llm_configs
        ]
        nd_client.chat.completions.model_select.assert_called_with(
            messages=[{"role": "user", "content": prompt}], tradeoff="cost"
        )

        chats = [{"role": "user", "content": "Hello, world!"}]
        actual_select = not_diamond_runnable._model_select(chats)
        assert str(actual_select) in [
            _nd_provider_to_langchain_provider(str(config)) for config in llm_configs
        ]
        nd_client.chat.completions.model_select.assert_called_with(
            messages=chats, tradeoff="cost"
        )

    @pytest.mark.asyncio
    async def test_amodel_select(
        self,
        not_diamond_runnable: NotDiamondRunnable,
        llm_configs: List,
        nd_client: Any,
    ) -> None:
        prompt = "Hello, world!"
        actual_select = await not_diamond_runnable._amodel_select(prompt)
        assert str(actual_select) in [
            _nd_provider_to_langchain_provider(str(config)) for config in llm_configs
        ]
        nd_client.chat.completions.amodel_select.assert_called_with(
            messages=[{"role": "user", "content": prompt}], tradeoff="cost"
        )

        chats = [{"role": "user", "content": "Hello, world!"}]
        actual_select = await not_diamond_runnable._amodel_select(chats)
        assert str(actual_select) in [
            _nd_provider_to_langchain_provider(str(config)) for config in llm_configs
        ]
        nd_client.chat.completions.amodel_select.assert_called_with(
            messages=chats, tradeoff="cost"
        )


class TestNotDiamondRoutedRunnable:
    def test_invoke(
        self, not_diamond_routed_runnable: NotDiamondRoutedRunnable, nd_client: Any
    ) -> None:
        prompt = "Hello, world!"
        not_diamond_routed_runnable.invoke(prompt)
        assert (
            not_diamond_routed_runnable._configurable_model.invoke.called  # type: ignore[attr-defined]
        ), f"{not_diamond_routed_runnable._configurable_model}"
        nd_client.chat.completions.model_select.assert_called_with(
            messages=[{"role": "user", "content": prompt}], tradeoff="cost"
        )

        # Check the call list
        call_list = (
            not_diamond_routed_runnable._configurable_model.invoke.call_args_list  # type: ignore[attr-defined]
        )
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == "Hello, world!"

    def test_stream(
        self, not_diamond_routed_runnable: NotDiamondRoutedRunnable, nd_client: Any
    ) -> None:
        prompt = "Hello, world!"
        for result in not_diamond_routed_runnable.stream(prompt):
            assert result is not None
        assert (
            not_diamond_routed_runnable._configurable_model.stream.called  # type: ignore[attr-defined]
        ), f"{not_diamond_routed_runnable._configurable_model}"
        nd_client.chat.completions.model_select.assert_called_with(
            messages=[{"role": "user", "content": prompt}], tradeoff="cost"
        )

    def test_batch(
        self, not_diamond_routed_runnable: NotDiamondRoutedRunnable, nd_client: Any
    ) -> None:
        prompts = ["Hello, world!", "How are you today?"]
        not_diamond_routed_runnable.batch(prompts)
        assert (
            not_diamond_routed_runnable._configurable_model.batch.called  # type: ignore[attr-defined]
        ), f"{not_diamond_routed_runnable._configurable_model}"
        nd_client_call_list = nd_client.chat.completions.model_select.call_args_list
        for call, prompt in zip(nd_client_call_list, prompts):
            args, kwargs = call
            assert kwargs["messages"] == [{"role": "user", "content": prompt}]
            assert kwargs["tradeoff"] == "cost"

        # Check the call list
        call_list = (
            not_diamond_routed_runnable._configurable_model.batch.call_args_list  # type: ignore[attr-defined]
        )
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == prompts

    @pytest.mark.asyncio
    async def test_ainvoke(
        self, not_diamond_routed_runnable: NotDiamondRoutedRunnable, nd_client: Any
    ) -> None:
        prompt = "Hello, world!"
        await not_diamond_routed_runnable.ainvoke(prompt)
        assert (
            not_diamond_routed_runnable._configurable_model.ainvoke.called  # type: ignore[attr-defined]
        ), f"{not_diamond_routed_runnable._configurable_model}"
        nd_client.chat.completions.amodel_select.assert_called_with(
            messages=[{"role": "user", "content": prompt}], tradeoff="cost"
        )

        # Check the call list
        call_list = (
            not_diamond_routed_runnable._configurable_model.ainvoke.call_args_list  # type: ignore[attr-defined]
        )
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_astream(
        self, not_diamond_routed_runnable: NotDiamondRoutedRunnable, nd_client: Any
    ) -> None:
        prompt = "Hello, world!"
        async for result in not_diamond_routed_runnable.astream(prompt):
            assert result is not None
        assert (
            not_diamond_routed_runnable._configurable_model.astream.called  # type: ignore[attr-defined]
        ), f"{not_diamond_routed_runnable._configurable_model}"
        nd_client.chat.completions.amodel_select.assert_called_with(
            messages=[{"role": "user", "content": prompt}], tradeoff="cost"
        )

    @pytest.mark.asyncio
    async def test_abatch(
        self, not_diamond_routed_runnable: NotDiamondRoutedRunnable, nd_client: Any
    ) -> None:
        prompts = ["Hello, world!", "How are you today?"]
        await not_diamond_routed_runnable.abatch(prompts)
        assert (
            not_diamond_routed_runnable._configurable_model.abatch.called  # type: ignore[attr-defined]
        ), f"{not_diamond_routed_runnable._configurable_model}"
        nd_client_call_list = nd_client.chat.completions.amodel_select.call_args_list
        for call, prompt in zip(nd_client_call_list, prompts):
            args, kwargs = call
            assert kwargs["messages"] == [{"role": "user", "content": prompt}]
            assert kwargs["tradeoff"] == "cost"

        # Check the call list
        call_list = (
            not_diamond_routed_runnable._configurable_model.abatch.call_args_list  # type: ignore[attr-defined]
        )
        assert len(call_list) == 1
        args, kwargs = call_list[0]
        assert args[0] == prompts

    def test_invokable_mock(self) -> None:
        target_model = "openai/gpt-4o"

        nd_client = MagicMock(
            llm_configs=[target_model],
            api_key="",
            default=target_model,
        )
        nd_client.chat.completions.model_select = MagicMock(
            return_value=(uuid.uuid4(), target_model)
        )

        mock_client = MagicMock(spec=LLM)

        with patch(
            "langchain_community.utilities.notdiamond.init_chat_model", autospec=True
        ) as mock_method:
            mock_method.return_value = mock_client
            with patch(
                "langchain_community.utilities.notdiamond.LLMConfig"
            ) as mock_llm_config:
                mock_llm_config.from_string.return_value = MagicMock(provider="openai")
                runnable = NotDiamondRoutedRunnable(nd_client=nd_client)
            runnable._ndrunnable.client.chat.completions.model_select.return_value = (
                uuid.uuid4(),
                target_model,
            )
            runnable.invoke("Test prompt")
            assert (
                mock_client.invoke.called  # type: ignore[attr-defined]
            ), f"{mock_client}"

        mock_client.reset_mock()

        with patch(
            "langchain_community.utilities.notdiamond.init_chat_model", autospec=True
        ) as mock_method:
            mock_method.return_value = mock_client
            with patch(
                "langchain_community.utilities.notdiamond.LLMConfig"
            ) as mock_llm_config:
                mock_llm_config.from_string.return_value = MagicMock(provider="openai")
                runnable = NotDiamondRoutedRunnable(
                    nd_api_key="sk-...", nd_llm_configs=[target_model]
                )
            runnable._ndrunnable.client.chat.completions.model_select.return_value = (
                uuid.uuid4(),
                target_model,
            )
            runnable.invoke("Test prompt")
            assert (
                mock_client.invoke.called  # type: ignore[attr-defined]
            ), f"{mock_client}"

    def test_init_perplexity(self) -> None:
        target_model = "perplexity/llama-3.1-sonar-large-128k-online"
        nd_client = MagicMock(
            llm_configs=[target_model],
            api_key="",
            default=target_model,
        )
        nd_client.chat.completions.model_select = MagicMock(
            return_value=(uuid.uuid4(), target_model)
        )
