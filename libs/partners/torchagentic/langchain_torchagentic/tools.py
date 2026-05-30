"""Tool wrapping torchagentic differentiable planners."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class TorchAgenticPlannerInput(BaseModel):
    """Input schema for TorchAgenticPlannerTool."""

    task: str = Field(description="A description of the planning task or goal.")
    num_states: int | None = Field(
        default=None,
        description="Number of abstract states for the planner. Defaults to the value set at initialization.",
    )
    num_actions: int | None = Field(
        default=None,
        description="Number of abstract actions for the planner. Defaults to the value set at initialization.",
    )
    planner_type: Literal["vi", "mcts"] | None = Field(
        default=None,
        description="Type of planner: 'vi' for value iteration, 'mcts' for Monte Carlo tree search. Defaults to the value set at initialization.",
    )


class TorchAgenticPlannerTool(BaseTool):
    r"""Tool that runs a differentiable planner using torchagentic primitives.

    Uses value iteration or MCTS on randomly initialized reward and transition
    models to produce a plan summary (top-ranked states or actions).

    Setup:
        Install ``langchain-torchagentic``.

        .. code-block:: bash

            pip install -U langchain-torchagentic

    Instantiation:
        .. code-block:: python

            from langchain_torchagentic import TorchAgenticPlannerTool

            tool = TorchAgenticPlannerTool(
                num_states=64,
                num_actions=8,
                planner_type="vi",
            )

    Invocation with args:
        .. code-block:: python

            tool.invoke({"task": "Find optimal route from A to B"})

        .. code-block:: python

            "[TorchAgentic | VI | top states: [3, 12, 13] | states: 64 | actions: 8 | task: Find optimal route from A to B]"
    """  # noqa: E501

    name: str = "torchagentic_planner"
    description: str = (
        "Runs a differentiable planner (value iteration or MCTS) on randomly "
        "initialized reward and transition models and returns a plan summary "
        "of top-ranked states or actions."
    )
    args_schema: type[BaseModel] = TorchAgenticPlannerInput

    num_states: int = Field(
        default=64,
        description="Number of abstract states for the planner.",
    )
    num_actions: int = Field(
        default=8,
        description="Number of abstract actions for the planner.",
    )
    planner_type: Literal["vi", "mcts"] = Field(
        default="vi",
        description="Type of planner: 'vi' for value iteration, 'mcts' for Monte Carlo tree search.",
    )
    gamma: float = Field(
        default=0.99,
        description="Discount factor for the planner.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from torchagentic.nn.planner import MCTSPlanner, ValueIteration
        except ImportError as e:
            raise ImportError(
                "TorchAgenticPlannerTool requires torchagentic. "
                "Install it with: pip install langchain-torchagentic"
            ) from e

        pt = self.planner_type
        if pt == "vi":
            self._planner = ValueIteration(
                num_states=self.num_states,
                num_actions=self.num_actions,
                gamma=self.gamma,
                num_iters=20,
            )
        elif pt == "mcts":
            self._planner = MCTSPlanner(
                num_simulations=20,
                c_puct=1.25,
                gamma=self.gamma,
            )
        else:
            raise ValueError(f"Unknown planner_type: {pt}")
        self._planner_type = pt

    def _run(
        self,
        task: str,
        num_states: int | None = None,
        num_actions: int | None = None,
        planner_type: Literal["vi", "mcts"] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        S = num_states or self.num_states
        A = num_actions or self.num_actions
        pt = planner_type or self._planner_type

        with torch.no_grad():
            reward = torch.randn(1, S, A)
            kernel = torch.randn(1, S, A, S)
            kernel = F.softmax(kernel.reshape(1, S * A, S), dim=-1).reshape(1, S, A, S)

            if pt == "vi":
                from torchagentic.nn.planner import ValueIteration

                planner = ValueIteration(
                    num_states=S,
                    num_actions=A,
                    gamma=self.gamma,
                    num_iters=20,
                )
                values, q_values = planner(reward, kernel)
                best_q = q_values.max(dim=-1).values.squeeze(0)
                top_ids = best_q.topk(min(3, S)).indices.tolist()
                plan_summary = f"top states: {top_ids}"
            else:
                from torchagentic.nn.planner import MCTSPlanner

                prior = torch.randn(1, A)
                value = torch.randn(1)
                planner = MCTSPlanner(
                    num_simulations=20,
                    c_puct=1.25,
                    gamma=self.gamma,
                )
                probs, _ = planner(prior, value)
                top_ids = probs.topk(min(3, A), dim=-1).indices.squeeze(0).tolist()
                plan_summary = f"top actions: {top_ids}"

        return (
            f"[TorchAgentic | {pt.upper()} | {plan_summary} | "
            f"states: {S} | actions: {A} | task: {task[:80]}]"
        )
