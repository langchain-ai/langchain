"""Unit tests for MetricsMiddleware."""

import time
from datetime import datetime, timezone
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware.metrics import (
    AgentRunMetrics,
    CallbackMetricsExporter,
    InMemoryMetricsExporter,
    MetricsMiddleware,
    MetricsMultiExporter,
    ModelCallMetrics,
    ToolCallMetrics,
)
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def simple_tool(value: str) -> str:
    """A simple test tool."""
    return f"Result: {value}"


@tool
def slow_tool(value: str) -> str:
    """A slow test tool."""
    time.sleep(0.05)
    return f"Slow result: {value}"


@tool
def failing_tool(value: str) -> str:
    """A tool that always fails."""
    msg = "Tool error"
    raise ValueError(msg)


class FailingModel(GenericFakeChatModel):
    """Model that always fails."""

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = "Model failed"
        raise ValueError(msg)


class ModelWithTokenUsage(GenericFakeChatModel):
    """Model that returns token usage metadata."""

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        if result.generations and result.generations[0]:
            msg = result.generations[0].message
            if isinstance(msg, AIMessage):
                msg.usage_metadata = {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                }
        return result


class TestInMemoryMetricsExporterSync:
    """Test InMemoryMetricsExporter sync functionality."""

    def test_stores_model_calls_sync(self) -> None:
        """Test that model call metrics are stored correctly (sync)."""
        exporter = InMemoryMetricsExporter()
        metrics = ModelCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            latency_ms=100.0,
            model_name="gpt-4",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
        exporter.export_model_call(metrics)

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].latency_ms == 100.0
        assert exporter.model_calls[0].total_tokens == 30
        assert exporter.model_calls[0].model_name == "gpt-4"

    def test_stores_tool_calls_sync(self) -> None:
        """Test that tool call metrics are stored correctly (sync)."""
        exporter = InMemoryMetricsExporter()
        metrics = ToolCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            tool_name="test_tool",
            latency_ms=50.0,
        )
        exporter.export_tool_call(metrics)

        assert len(exporter.tool_calls) == 1
        assert exporter.tool_calls[0].tool_name == "test_tool"
        assert exporter.tool_calls[0].latency_ms == 50.0

    def test_stores_run_metrics_sync(self) -> None:
        """Test that run metrics are stored correctly (sync)."""
        exporter = InMemoryMetricsExporter()
        run = AgentRunMetrics(
            run_id="test-run",
            start_time=datetime.now(tz=timezone.utc),
            end_time=datetime.now(tz=timezone.utc),
        )
        exporter.export_run_complete(run)

        assert len(exporter.runs) == 1
        assert exporter.runs[0].run_id == "test-run"

    def test_clear(self) -> None:
        """Test clearing all stored metrics."""
        exporter = InMemoryMetricsExporter()
        exporter.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )
        exporter.export_tool_call(
            ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="tool", latency_ms=50.0)
        )
        exporter.export_run_complete(AgentRunMetrics("id", datetime.now(tz=timezone.utc)))
        exporter.clear()

        assert len(exporter.model_calls) == 0
        assert len(exporter.tool_calls) == 0
        assert len(exporter.runs) == 0

    def test_total_tokens_property(self) -> None:
        """Test total_tokens aggregation property."""
        exporter = InMemoryMetricsExporter()
        exporter.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0, total_tokens=50)
        )
        exporter.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0, total_tokens=100)
        )
        exporter.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0, total_tokens=None)
        )

        assert exporter.total_tokens == 150

    def test_average_latency_properties(self) -> None:
        """Test average latency calculation properties."""
        exporter = InMemoryMetricsExporter()
        exporter.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )
        exporter.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=200.0)
        )
        exporter.export_tool_call(
            ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="tool1", latency_ms=50.0)
        )
        exporter.export_tool_call(
            ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="tool2", latency_ms=150.0)
        )

        assert exporter.average_model_latency_ms == 150.0
        assert exporter.average_tool_latency_ms == 100.0

    def test_average_latency_empty(self) -> None:
        """Test average latency with no data."""
        exporter = InMemoryMetricsExporter()

        assert exporter.average_model_latency_ms == 0.0
        assert exporter.average_tool_latency_ms == 0.0


class TestInMemoryMetricsExporterAsync:
    """Test InMemoryMetricsExporter async functionality."""

    @pytest.mark.asyncio
    async def test_stores_model_calls_async(self) -> None:
        """Test that model call metrics are stored correctly (async)."""
        exporter = InMemoryMetricsExporter()
        metrics = ModelCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            latency_ms=100.0,
            model_name="gpt-4",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
        await exporter.aexport_model_call(metrics)

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].latency_ms == 100.0
        assert exporter.model_calls[0].total_tokens == 30
        assert exporter.model_calls[0].model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_stores_tool_calls_async(self) -> None:
        """Test that tool call metrics are stored correctly (async)."""
        exporter = InMemoryMetricsExporter()
        metrics = ToolCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            tool_name="test_tool",
            latency_ms=50.0,
        )
        await exporter.aexport_tool_call(metrics)

        assert len(exporter.tool_calls) == 1
        assert exporter.tool_calls[0].tool_name == "test_tool"
        assert exporter.tool_calls[0].latency_ms == 50.0

    @pytest.mark.asyncio
    async def test_stores_run_metrics_async(self) -> None:
        """Test that run metrics are stored correctly (async)."""
        exporter = InMemoryMetricsExporter()
        run = AgentRunMetrics(
            run_id="test-run",
            start_time=datetime.now(tz=timezone.utc),
            end_time=datetime.now(tz=timezone.utc),
        )
        await exporter.aexport_run_complete(run)

        assert len(exporter.runs) == 1
        assert exporter.runs[0].run_id == "test-run"


class TestCallbackMetricsExporterSync:
    """Test CallbackMetricsExporter sync functionality."""

    def test_invokes_model_call_callback_sync(self) -> None:
        """Test that sync model call callback is invoked."""
        model_calls: list[ModelCallMetrics] = []

        def on_model_call(m: ModelCallMetrics) -> None:
            model_calls.append(m)

        exporter = CallbackMetricsExporter(on_model_call=on_model_call)

        metrics = ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        exporter.export_model_call(metrics)

        assert len(model_calls) == 1
        assert model_calls[0].latency_ms == 100.0

    def test_invokes_tool_call_callback_sync(self) -> None:
        """Test that sync tool call callback is invoked."""
        tool_calls: list[ToolCallMetrics] = []

        def on_tool_call(t: ToolCallMetrics) -> None:
            tool_calls.append(t)

        exporter = CallbackMetricsExporter(on_tool_call=on_tool_call)

        metrics = ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="my_tool", latency_ms=50.0)
        exporter.export_tool_call(metrics)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "my_tool"

    def test_invokes_run_complete_callback_sync(self) -> None:
        """Test that sync run complete callback is invoked."""
        runs: list[AgentRunMetrics] = []

        def on_run_complete(r: AgentRunMetrics) -> None:
            runs.append(r)

        exporter = CallbackMetricsExporter(on_run_complete=on_run_complete)

        metrics = AgentRunMetrics("run-123", datetime.now(tz=timezone.utc))
        exporter.export_run_complete(metrics)

        assert len(runs) == 1
        assert runs[0].run_id == "run-123"

    def test_handles_none_callbacks_sync(self) -> None:
        """Test that None callbacks don't cause errors (sync)."""
        exporter = CallbackMetricsExporter()

        exporter.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )
        exporter.export_tool_call(
            ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="tool", latency_ms=50.0)
        )
        exporter.export_run_complete(AgentRunMetrics("id", datetime.now(tz=timezone.utc)))


class TestCallbackMetricsExporterAsync:
    """Test CallbackMetricsExporter async functionality."""

    @pytest.mark.asyncio
    async def test_invokes_async_model_call_callback(self) -> None:
        """Test that async model call callback is invoked."""
        model_calls: list[ModelCallMetrics] = []

        async def aon_model_call(m: ModelCallMetrics) -> None:
            model_calls.append(m)

        exporter = CallbackMetricsExporter(aon_model_call=aon_model_call)

        metrics = ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        await exporter.aexport_model_call(metrics)

        assert len(model_calls) == 1
        assert model_calls[0].latency_ms == 100.0

    @pytest.mark.asyncio
    async def test_invokes_async_tool_call_callback(self) -> None:
        """Test that async tool call callback is invoked."""
        tool_calls: list[ToolCallMetrics] = []

        async def aon_tool_call(t: ToolCallMetrics) -> None:
            tool_calls.append(t)

        exporter = CallbackMetricsExporter(aon_tool_call=aon_tool_call)

        metrics = ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="my_tool", latency_ms=50.0)
        await exporter.aexport_tool_call(metrics)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "my_tool"

    @pytest.mark.asyncio
    async def test_invokes_async_run_complete_callback(self) -> None:
        """Test that async run complete callback is invoked."""
        runs: list[AgentRunMetrics] = []

        async def aon_run_complete(r: AgentRunMetrics) -> None:
            runs.append(r)

        exporter = CallbackMetricsExporter(aon_run_complete=aon_run_complete)

        metrics = AgentRunMetrics("run-123", datetime.now(tz=timezone.utc))
        await exporter.aexport_run_complete(metrics)

        assert len(runs) == 1
        assert runs[0].run_id == "run-123"

    @pytest.mark.asyncio
    async def test_async_falls_back_to_sync_callback(self) -> None:
        """Test that async methods fall back to sync callbacks if no async provided."""
        model_calls: list[ModelCallMetrics] = []

        def on_model_call(m: ModelCallMetrics) -> None:
            model_calls.append(m)

        exporter = CallbackMetricsExporter(on_model_call=on_model_call)

        metrics = ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        await exporter.aexport_model_call(metrics)

        assert len(model_calls) == 1

    @pytest.mark.asyncio
    async def test_handles_none_callbacks_async(self) -> None:
        """Test that None callbacks don't cause errors (async)."""
        exporter = CallbackMetricsExporter()

        await exporter.aexport_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )
        await exporter.aexport_tool_call(
            ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="tool", latency_ms=50.0)
        )
        await exporter.aexport_run_complete(AgentRunMetrics("id", datetime.now(tz=timezone.utc)))


class TestMetricsMultiExporterSync:
    """Test MetricsMultiExporter sync functionality."""

    def test_forwards_to_all_exporters_sync(self) -> None:
        """Test that metrics are forwarded to all child exporters (sync)."""
        exporter1 = InMemoryMetricsExporter()
        exporter2 = InMemoryMetricsExporter()

        exporters = MetricsMultiExporter([exporter1, exporter2])

        exporters.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )
        exporters.export_tool_call(
            ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="tool", latency_ms=50.0)
        )
        exporters.export_run_complete(AgentRunMetrics("id", datetime.now(tz=timezone.utc)))

        assert len(exporter1.model_calls) == 1
        assert len(exporter2.model_calls) == 1
        assert len(exporter1.tool_calls) == 1
        assert len(exporter2.tool_calls) == 1
        assert len(exporter1.runs) == 1
        assert len(exporter2.runs) == 1

    def test_empty_exporters_sync(self) -> None:
        """Test that empty exporters doesn't error (sync)."""
        exporters = MetricsMultiExporter([])

        exporters.export_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )


class TestMetricsMultiExporterAsync:
    """Test MetricsMultiExporter async functionality."""

    @pytest.mark.asyncio
    async def test_forwards_to_all_exporters_async(self) -> None:
        """Test that metrics are forwarded to all child exporters (async)."""
        exporter1 = InMemoryMetricsExporter()
        exporter2 = InMemoryMetricsExporter()

        exporters = MetricsMultiExporter([exporter1, exporter2])

        await exporters.aexport_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )
        await exporters.aexport_tool_call(
            ToolCallMetrics(timestamp=datetime.now(tz=timezone.utc), tool_name="tool", latency_ms=50.0)
        )
        await exporters.aexport_run_complete(AgentRunMetrics("id", datetime.now(tz=timezone.utc)))

        assert len(exporter1.model_calls) == 1
        assert len(exporter2.model_calls) == 1
        assert len(exporter1.tool_calls) == 1
        assert len(exporter2.tool_calls) == 1
        assert len(exporter1.runs) == 1
        assert len(exporter2.runs) == 1

    @pytest.mark.asyncio
    async def test_empty_exporters_async(self) -> None:
        """Test that empty exporters doesn't error (async)."""
        exporters = MetricsMultiExporter([])

        await exporters.aexport_model_call(
            ModelCallMetrics(timestamp=datetime.now(tz=timezone.utc), latency_ms=100.0)
        )


class TestAgentRunMetrics:
    """Test AgentRunMetrics properties."""

    def test_total_latency(self) -> None:
        """Test total_latency_ms calculation."""
        now = datetime.now(tz=timezone.utc)
        run = AgentRunMetrics(
            run_id="test",
            start_time=now,
            model_calls=[
                ModelCallMetrics(timestamp=now, latency_ms=100.0),
                ModelCallMetrics(timestamp=now, latency_ms=200.0),
            ],
            tool_calls=[
                ToolCallMetrics(timestamp=now, tool_name="tool1", latency_ms=50.0),
                ToolCallMetrics(timestamp=now, tool_name="tool2", latency_ms=50.0),
            ],
        )

        assert run.total_latency_ms == 400.0

    def test_total_tokens(self) -> None:
        """Test total_tokens calculation."""
        now = datetime.now(tz=timezone.utc)
        run = AgentRunMetrics(
            run_id="test",
            start_time=now,
            model_calls=[
                ModelCallMetrics(timestamp=now, latency_ms=100.0, total_tokens=50),
                ModelCallMetrics(timestamp=now, latency_ms=200.0, total_tokens=100),
                ModelCallMetrics(timestamp=now, latency_ms=200.0, total_tokens=None),
            ],
        )

        assert run.total_tokens == 150

    def test_input_output_tokens(self) -> None:
        """Test input/output token calculations."""
        now = datetime.now(tz=timezone.utc)
        run = AgentRunMetrics(
            run_id="test",
            start_time=now,
            model_calls=[
                ModelCallMetrics(timestamp=now, latency_ms=100.0, input_tokens=80, output_tokens=20),
                ModelCallMetrics(timestamp=now, latency_ms=100.0, input_tokens=60, output_tokens=40),
            ],
        )

        assert run.total_input_tokens == 140
        assert run.total_output_tokens == 60

    def test_call_counts(self) -> None:
        """Test model_call_count and tool_call_count properties."""
        now = datetime.now(tz=timezone.utc)
        run = AgentRunMetrics(
            run_id="test",
            start_time=now,
            model_calls=[
                ModelCallMetrics(timestamp=now, latency_ms=100.0),
                ModelCallMetrics(timestamp=now, latency_ms=200.0),
            ],
            tool_calls=[
                ToolCallMetrics(timestamp=now, tool_name="tool1", latency_ms=50.0),
            ],
        )

        assert run.model_call_count == 2
        assert run.tool_call_count == 1

    def test_success_rates(self) -> None:
        """Test success rate calculations."""
        now = datetime.now(tz=timezone.utc)
        test_error = ValueError("test failure")
        run = AgentRunMetrics(
            run_id="test",
            start_time=now,
            model_calls=[
                ModelCallMetrics(timestamp=now, latency_ms=100.0),
                ModelCallMetrics(timestamp=now, latency_ms=200.0, error=test_error),
            ],
            tool_calls=[
                ToolCallMetrics(timestamp=now, tool_name="tool1", latency_ms=50.0),
                ToolCallMetrics(timestamp=now, tool_name="tool2", latency_ms=50.0),
                ToolCallMetrics(timestamp=now, tool_name="tool3", latency_ms=50.0, error=test_error),
            ],
        )

        assert run.model_success_rate == 0.5
        assert run.tool_success_rate == pytest.approx(2 / 3)

    def test_success_rates_empty(self) -> None:
        """Test success rates with no calls."""
        run = AgentRunMetrics(
            run_id="test",
            start_time=datetime.now(tz=timezone.utc),
        )

        assert run.model_success_rate == 1.0
        assert run.tool_success_rate == 1.0


class TestMetricsMiddlewareSync:
    """Test middleware sync invocation."""

    def test_tracks_model_call_latency_sync(self) -> None:
        """Test that model call latency is tracked (sync)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        agent.invoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].latency_ms > 0
        assert exporter.model_calls[0].success is True
        assert exporter.model_calls[0].error is None

    def test_tracks_model_call_failure_sync(self) -> None:
        """Test that model failures are tracked (sync)."""
        exporter = InMemoryMetricsExporter()

        agent = create_agent(
            model=FailingModel(messages=iter([])),
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        with pytest.raises(ValueError, match="Model failed"):
            agent.invoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].success is False
        assert isinstance(exporter.model_calls[0].error, ValueError)

    def test_tracks_token_usage_sync(self) -> None:
        """Test that token usage is extracted from responses (sync)."""
        exporter = InMemoryMetricsExporter()
        model = ModelWithTokenUsage(messages=iter([AIMessage(content="Hello")]))

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        agent.invoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].input_tokens == 100
        assert exporter.model_calls[0].output_tokens == 50
        assert exporter.model_calls[0].total_tokens == 150

    def test_tracks_tool_call_latency_sync(self) -> None:
        """Test that tool call latency is tracked (sync)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="simple_tool", args={"value": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        agent.invoke({"messages": [HumanMessage("Use the tool")]})

        assert len(exporter.tool_calls) == 1
        assert exporter.tool_calls[0].tool_name == "simple_tool"
        assert exporter.tool_calls[0].latency_ms > 0
        assert exporter.tool_calls[0].success is True

    def test_exports_run_metrics_sync(self) -> None:
        """Test that run metrics are exported on completion (sync)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="simple_tool", args={"value": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        agent.invoke({"messages": [HumanMessage("Use the tool")]})

        assert len(exporter.runs) == 1
        run = exporter.runs[0]
        assert run.run_id is not None
        assert run.start_time is not None
        assert run.end_time is not None
        assert run.model_call_count == 2
        assert run.tool_call_count == 1

    def test_works_without_exporter_sync(self) -> None:
        """Test that middleware works when no exporter is provided (sync)."""
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware()],
        )

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result


class TestMetricsMiddlewareAsync:
    """Test middleware async invocation."""

    @pytest.mark.asyncio
    async def test_tracks_model_call_latency_async(self) -> None:
        """Test that model call latency is tracked (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].latency_ms > 0
        assert exporter.model_calls[0].success is True
        assert exporter.model_calls[0].error is None

    @pytest.mark.asyncio
    async def test_tracks_model_call_failure_async(self) -> None:
        """Test that model failures are tracked (async)."""
        exporter = InMemoryMetricsExporter()

        agent = create_agent(
            model=FailingModel(messages=iter([])),
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        with pytest.raises(ValueError, match="Model failed"):
            await agent.ainvoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].success is False
        assert isinstance(exporter.model_calls[0].error, ValueError)

    @pytest.mark.asyncio
    async def test_tracks_token_usage_async(self) -> None:
        """Test that token usage is extracted from responses (async)."""
        exporter = InMemoryMetricsExporter()
        model = ModelWithTokenUsage(messages=iter([AIMessage(content="Hello")]))

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].input_tokens == 100
        assert exporter.model_calls[0].output_tokens == 50
        assert exporter.model_calls[0].total_tokens == 150

    @pytest.mark.asyncio
    async def test_tracks_tool_call_latency_async(self) -> None:
        """Test that tool call latency is tracked (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="simple_tool", args={"value": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Use the tool")]})

        assert len(exporter.tool_calls) == 1
        assert exporter.tool_calls[0].tool_name == "simple_tool"
        assert exporter.tool_calls[0].latency_ms > 0
        assert exporter.tool_calls[0].success is True

    @pytest.mark.asyncio
    async def test_tracks_slow_tool_async(self) -> None:
        """Test that slow tool latency is measured correctly (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="slow_tool", args={"value": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[slow_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Use the tool")]})

        assert exporter.tool_calls[0].latency_ms >= 50

    @pytest.mark.asyncio
    async def test_tracks_multiple_tool_calls_async(self) -> None:
        """Test tracking multiple tool calls (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(name="simple_tool", args={"value": "first"}, id="1"),
                    ToolCall(name="simple_tool", args={"value": "second"}, id="2"),
                ],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Use tools")]})

        assert len(exporter.tool_calls) == 2
        assert all(t.tool_name == "simple_tool" for t in exporter.tool_calls)

    @pytest.mark.asyncio
    async def test_exports_run_metrics_async(self) -> None:
        """Test that run metrics are exported on completion (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="simple_tool", args={"value": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Use the tool")]})

        assert len(exporter.runs) == 1
        run = exporter.runs[0]
        assert run.run_id is not None
        assert run.start_time is not None
        assert run.end_time is not None
        assert run.model_call_count == 2
        assert run.tool_call_count == 1

    @pytest.mark.asyncio
    async def test_run_metrics_aggregation_async(self) -> None:
        """Test that run metrics contain aggregated data (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="simple_tool", args={"value": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Use the tool")]})

        run = exporter.runs[0]
        assert run.total_latency_ms > 0
        assert len(run.model_calls) == 2
        assert len(run.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_works_without_exporter_async(self) -> None:
        """Test that middleware works when no exporter is provided (async)."""
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware()],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result


class TestMetricsMiddlewareIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_agent_run_with_metrics_async(self) -> None:
        """Test complete agent run with all metrics collected (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(name="simple_tool", args={"value": "first"}, id="1"),
                ],
                [
                    ToolCall(name="simple_tool", args={"value": "second"}, id="2"),
                ],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Use tools multiple times")]})

        assert len(exporter.model_calls) == 3
        assert len(exporter.tool_calls) == 2
        assert len(exporter.runs) == 1

        run = exporter.runs[0]
        assert run.model_call_count == 3
        assert run.tool_call_count == 2
        assert run.total_latency_ms > 0

    def test_full_agent_run_with_metrics_sync(self) -> None:
        """Test complete agent run with all metrics collected (sync)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(name="simple_tool", args={"value": "first"}, id="1"),
                ],
                [
                    ToolCall(name="simple_tool", args={"value": "second"}, id="2"),
                ],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[simple_tool],
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        agent.invoke({"messages": [HumanMessage("Use tools multiple times")]})

        assert len(exporter.model_calls) == 3
        assert len(exporter.tool_calls) == 2
        assert len(exporter.runs) == 1

        run = exporter.runs[0]
        assert run.model_call_count == 3
        assert run.tool_call_count == 2
        assert run.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_combined_with_other_middleware(self) -> None:
        """Test that MetricsMiddleware works with other middleware."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[
                MetricsMiddleware(exporter=exporter),
                ModelRetryMiddleware(max_retries=1),
            ],
        )

        await agent.ainvoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert len(exporter.runs) == 1


class TestTraceCorrelation:
    """Test trace correlation features (OpenTelemetry and LangSmith)."""

    def test_model_call_metrics_has_trace_fields(self) -> None:
        """Test that ModelCallMetrics has trace correlation fields."""
        metrics = ModelCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            latency_ms=100.0,
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            run_id="ls-run-123",
        )

        assert metrics.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert metrics.span_id == "b7ad6b7169203331"
        assert metrics.run_id == "ls-run-123"

    def test_tool_call_metrics_has_trace_fields(self) -> None:
        """Test that ToolCallMetrics has trace correlation fields."""
        metrics = ToolCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            tool_name="my_tool",
            latency_ms=50.0,
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            run_id="ls-run-456",
        )

        assert metrics.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert metrics.span_id == "b7ad6b7169203331"
        assert metrics.run_id == "ls-run-456"

    def test_trace_fields_default_to_none(self) -> None:
        """Test that trace fields default to None when not provided."""
        model_metrics = ModelCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            latency_ms=100.0,
        )
        tool_metrics = ToolCallMetrics(
            timestamp=datetime.now(tz=timezone.utc),
            tool_name="test",
            latency_ms=50.0,
        )

        assert model_metrics.trace_id is None
        assert model_metrics.span_id is None
        assert model_metrics.run_id is None
        assert tool_metrics.trace_id is None
        assert tool_metrics.span_id is None
        assert tool_metrics.run_id is None

    @pytest.mark.asyncio
    async def test_metrics_collected_without_tracing_async(self) -> None:
        """Test that metrics work without any tracing configured (async)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        await agent.ainvoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].trace_id is None
        assert exporter.model_calls[0].span_id is None
        assert exporter.model_calls[0].run_id is None

    def test_metrics_collected_without_tracing_sync(self) -> None:
        """Test that metrics work without any tracing configured (sync)."""
        exporter = InMemoryMetricsExporter()
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[MetricsMiddleware(exporter=exporter)],
        )

        agent.invoke({"messages": [HumanMessage("Hello")]})

        assert len(exporter.model_calls) == 1
        assert exporter.model_calls[0].trace_id is None
        assert exporter.model_calls[0].span_id is None
        assert exporter.model_calls[0].run_id is None

    def test_get_run_id_extraction(self) -> None:
        """Test run_id extraction from various config formats."""
        middleware = MetricsMiddleware()

        class NoConfigRequest:
            pass

        assert middleware._get_run_id(NoConfigRequest()) is None

        class DirectRunIdRequest:
            config = {"run_id": "direct-run-123"}

        assert middleware._get_run_id(DirectRunIdRequest()) == "direct-run-123"

        class MockCallback:
            run_id = "callback-run-456"

        class CallbackRequest:
            config = {"callbacks": [MockCallback()]}

        assert middleware._get_run_id(CallbackRequest()) == "callback-run-456"

        class MockParentCallback:
            run_id = None
            parent_run_id = "parent-run-789"

        class ParentCallbackRequest:
            config = {"callbacks": [MockParentCallback()]}

        assert middleware._get_run_id(ParentCallbackRequest()) == "parent-run-789"

        class EmptyCallbacksRequest:
            config = {"callbacks": []}

        assert middleware._get_run_id(EmptyCallbacksRequest()) is None

        class NoneConfigRequest:
            config = None

        assert middleware._get_run_id(NoneConfigRequest()) is None

    def test_get_otel_context_without_active_span(self) -> None:
        """Test OpenTelemetry context extraction when no active span."""
        middleware = MetricsMiddleware()

        trace_id, span_id = middleware._get_otel_context()

        assert trace_id is None or isinstance(trace_id, str)
        assert span_id is None or isinstance(span_id, str)

    def test_build_trace_context_combines_sources(self) -> None:
        """Test that _build_trace_context combines OTel and run_id."""
        middleware = MetricsMiddleware()

        class RequestWithRunId:
            config = {"run_id": "combined-123"}

        ctx = middleware._build_trace_context(RequestWithRunId())

        assert ctx.run_id == "combined-123"
        assert ctx.trace_id is None or isinstance(ctx.trace_id, str)
        assert ctx.span_id is None or isinstance(ctx.span_id, str)

    def test_trace_context_namedtuple_fields(self) -> None:
        """Test that _TraceContext has expected fields."""
        from langchain.agents.middleware.metrics import _TraceContext

        ctx = _TraceContext(
            trace_id="trace123",
            span_id="span456",
            run_id="run789",
        )

        assert ctx.trace_id == "trace123"
        assert ctx.span_id == "span456"
        assert ctx.run_id == "run789"

        ctx_defaults = _TraceContext()
        assert ctx_defaults.trace_id is None
        assert ctx_defaults.span_id is None
        assert ctx_defaults.run_id is None
