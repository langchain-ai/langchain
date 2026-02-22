from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents import create_agent


class _RecordingModel(BaseChatModel):
    """Model that records the config passed from the agent."""

    seen_config: dict[str, Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "recording-model"

    def _generate(  # type: ignore[override]
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    async def ainvoke(  # type: ignore[override]
        self,
        model_input: Any,
        config: dict[str, Any] | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        del model_input, stop, kwargs
        self.seen_config = config
        return AIMessage(content="done")


async def test_agent_model_calls_include_stable_tags() -> None:
    model = _RecordingModel()
    agent = create_agent(model=model, tools=[], name="my-agent")

    _ = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})

    assert model.seen_config is not None
    tags = model.seen_config.get("tags") or []
    assert "lc:agent_node:model" in tags
    assert "lc:agent_name:my-agent" in tags
