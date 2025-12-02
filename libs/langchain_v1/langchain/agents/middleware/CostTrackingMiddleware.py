"""成本追踪中间件 - 追踪模型调用的 token 使用和成本。

这个中间件会记录每次模型调用的输入/输出 token 数量，并计算成本。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Callable
from dataclasses import dataclass

from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


@dataclass
class TokenUsage:
    """Token 使用统计"""

    input_tokens: int = 0
    """输入 token 数量"""
    output_tokens: int = 0
    """输出 token 数量"""
    total_tokens: int = 0
    """总 token 数量"""

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """合并两个 TokenUsage"""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class CostInfo:
    """成本信息"""

    input_cost: float = 0.0
    """输入成本（美元）"""
    output_cost: float = 0.0
    """输出成本（美元）"""
    total_cost: float = 0.0
    """总成本（美元）"""

    def __add__(self, other: CostInfo) -> CostInfo:
        """合并两个 CostInfo"""
        return CostInfo(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
        )


class CostTrackingState(AgentState):
    """成本追踪状态模式"""

    thread_token_usage: NotRequired[Annotated[TokenUsage, PrivateStateAttr]]
    """线程级别的 token 使用统计"""
    run_token_usage: NotRequired[Annotated[TokenUsage, UntrackedValue, PrivateStateAttr]]
    """运行级别的 token 使用统计"""
    thread_cost: NotRequired[Annotated[CostInfo, PrivateStateAttr]]
    """线程级别的成本统计"""
    run_cost: NotRequired[Annotated[CostInfo, UntrackedValue, PrivateStateAttr]]
    """运行级别的成本统计"""


# 常见模型的定价（每 1M tokens，美元）
DEFAULT_PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


def _extract_model_name(model: Any) -> str:
    """从模型对象中提取模型名称"""
    if hasattr(model, "model_name"):
        return model.model_name
    if hasattr(model, "model"):
        return str(model.model)
    return str(model)


def _get_pricing(model_name: str, pricing_map: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """获取模型定价"""
    # 尝试精确匹配
    if model_name in pricing_map:
        return pricing_map[model_name]

    # 尝试部分匹配
    for key, value in pricing_map.items():
        if key.lower() in model_name.lower() or model_name.lower() in key.lower():
            return value

    return None


def _extract_token_usage(response: ModelResponse) -> TokenUsage:
    """从模型响应中提取 token 使用情况"""
    if not response.result:
        return TokenUsage()

    # 尝试从最后一个 AIMessage 中提取 token 使用信息
    last_message = response.result[-1]
    if isinstance(last_message, AIMessage) and hasattr(last_message, "usage_metadata"):
        usage = last_message.usage_metadata
        if usage:
            return TokenUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                total_tokens=(usage.input_tokens or 0) + (usage.output_tokens or 0),
            )

    return TokenUsage()


def _calculate_cost(token_usage: TokenUsage, pricing: dict[str, float] | None) -> CostInfo:
    """计算成本"""
    if not pricing:
        return CostInfo()

    input_cost = (token_usage.input_tokens / 1_000_000) * pricing.get("input", 0.0)
    output_cost = (token_usage.output_tokens / 1_000_000) * pricing.get("output", 0.0)

    return CostInfo(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


class CostTrackingMiddleware(AgentMiddleware[CostTrackingState, Any]):
    """成本追踪中间件 - 追踪模型调用的 token 使用和成本。

    这个中间件会记录每次模型调用的输入/输出 token 数量，并计算成本。
    支持线程级别和运行级别的统计。

    Example:
        from langchain.agents import create_agent
        from langchain.agents.middleware.cost_tracking import CostTrackingMiddleware

        # 使用默认定价
        cost_tracker = CostTrackingMiddleware()

        # 或自定义定价
        custom_pricing = {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "my-model": {"input": 1.0, "output": 2.0},
        }
        cost_tracker = CostTrackingMiddleware(
            pricing_map=custom_pricing,
            log_costs=True,  # 启用日志输出
        )

        agent = create_agent("openai:gpt-4o", middleware=[cost_tracker])

        result = agent.invoke({"messages": [HumanMessage("Hello")]})

        # 查看成本统计
        print(f"运行成本: ${result.get('run_cost', {}).get('total_cost', 0):.4f}")
        print(f"Token 使用: {result.get('run_token_usage', {})}")
            Args:
        pricing_map: 模型定价映射表，格式为 {model_name: {"input": price, "output": price}}
            如果为 None，使用默认定价表。
        log_costs: 是否在每次调用后打印成本信息。
        cost_callback: 可选的回调函数，在每次计算成本后调用。
            函数签名: callback(model_name: str, token_usage: TokenUsage, cost: CostInfo)
    """

    state_schema = CostTrackingState

    def __init__(
        self,
        *,
        pricing_map: dict[str, dict[str, float]] | None = None,
        log_costs: bool = False,
        cost_callback: Callable[[str, TokenUsage, CostInfo], None] | None = None,
    ) -> None:
        """初始化成本追踪中间件。

        Args:
            pricing_map: 模型定价映射表。如果为 None，使用默认定价。
            log_costs: 是否在每次调用后打印成本信息。
            cost_callback: 可选的成本回调函数。
        """
        super().__init__()
        self.pricing_map = pricing_map or DEFAULT_PRICING.copy()
        self.log_costs = log_costs
        self.cost_callback = cost_callback

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """拦截模型调用并追踪成本。

        Args:
            request: 模型请求。
            handler: 模型调用处理器。

        Returns:
            模型响应，包含 token 使用信息。
        """
        # 执行模型调用
        response = handler(request)

        # 提取 token 使用情况
        token_usage = _extract_token_usage(response)

        # 获取模型名称和定价
        model_name = _extract_model_name(request.model)
        pricing = _get_pricing(model_name, self.pricing_map)

        # 计算成本
        cost = _calculate_cost(token_usage, pricing)

        # 调用回调函数
        if self.cost_callback:
            try:
                self.cost_callback(model_name, token_usage, cost)
            except Exception:
                pass  # 忽略回调错误

        # 打印日志
        if self.log_costs:
            print(
                f"[CostTracking] Model: {model_name}\n"
                f"  Input tokens: {token_usage.input_tokens:,}\n"
                f"  Output tokens: {token_usage.output_tokens:,}\n"
                f"  Total tokens: {token_usage.total_tokens:,}\n"
                f"  Cost: ${cost.total_cost:.6f} (Input: ${cost.input_cost:.6f}, Output: ${cost.output_cost:.6f})"
            )

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """异步版本的模型调用拦截。

        Args:
            request: 模型请求。
            handler: 异步模型调用处理器。

        Returns:
            模型响应，包含 token 使用信息。
        """
        # 执行模型调用
        response = await handler(request)

        # 提取 token 使用情况
        token_usage = _extract_token_usage(response)

        # 获取模型名称和定价
        model_name = _extract_model_name(request.model)
        pricing = _get_pricing(model_name, self.pricing_map)

        # 计算成本
        cost = _calculate_cost(token_usage, pricing)

        # 调用回调函数
        if self.cost_callback:
            try:
                self.cost_callback(model_name, token_usage, cost)
            except Exception:
                pass  # 忽略回调错误

        # 打印日志
        if self.log_costs:
            print(
                f"[CostTracking] Model: {model_name}\n"
                f"  Input tokens: {token_usage.input_tokens:,}\n"
                f"  Output tokens: {token_usage.output_tokens:,}\n"
                f"  Total tokens: {token_usage.total_tokens:,}\n"
                f"  Cost: ${cost.total_cost:.6f} (Input: ${cost.input_cost:.6f}, Output: ${cost.output_cost:.6f})"
            )

        return response

    def after_model(self, state: CostTrackingState, runtime: Runtime) -> dict[str, Any] | None:
        """在模型调用后更新成本统计。

        Args:
            state: 当前状态。
            runtime: LangGraph 运行时。

        Returns:
            状态更新，包含累积的 token 使用和成本。
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # 获取最后一个 AI 消息
        last_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_message = msg
                break

        if not last_message or not hasattr(last_message, "usage_metadata"):
            return None

        usage = last_message.usage_metadata
        if not usage:
            return None

        # 提取本次调用的 token 使用
        current_usage = TokenUsage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            total_tokens=(usage.input_tokens or 0) + (usage.output_tokens or 0),
        )

        # 获取模型名称和定价
        model_name = _extract_model_name(runtime)  # 这里可能需要从 state 中获取
        # 简化处理：从 state 中获取模型信息
        # 实际实现中可能需要从 request 中获取

        # 计算本次成本
        pricing = _get_pricing(model_name, self.pricing_map)
        current_cost = _calculate_cost(current_usage, pricing)

        # 累积统计
        thread_usage = state.get("thread_token_usage", TokenUsage())
        run_usage = state.get("run_token_usage", TokenUsage())
        thread_cost = state.get("thread_cost", CostInfo())
        run_cost = state.get("run_cost", CostInfo())

        return {
            "thread_token_usage": thread_usage + current_usage,
            "run_token_usage": run_usage + current_usage,
            "thread_cost": thread_cost + current_cost,
            "run_cost": run_cost + current_cost,
        }