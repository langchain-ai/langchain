"""Benchmark harness: LLMToolSelectorMiddleware vs TsToolSelectorMiddleware.

Not part of the package (not linted/typed with the rest of langchain_typesafe,
not shipped) -- a standalone eval script for comparing tool-selection latency
and quality between the two middlewares, against a real 9-tool `deepagents`
roster (ls/read_file/write_file/edit_file/delete/glob/grep/execute/task).

Usage:
    cd libs/partners/typesafe
    OPENAI_BASE_URL=... OPENAI_API_KEY=... \\
    TYPESAFE_BASE_URL=... TYPESAFE_API_KEY=... \\
    [LANGSMITH_TRACING=true LANGSMITH_API_KEY=... LANGSMITH_PROJECT=...] \\
    uv run --with deepagents --with langchain-openai \\
        python scripts/eval_tool_selector_comparison.py

Requires an OpenAI-compatible endpoint serving `gpt-5.6-luna` (or edit MODEL
below) and a TypeSafe-compatible endpoint for the Jev classifier. LangSmith
tracing is optional -- only enabled if LANGSMITH_TRACING is set by the caller;
this script never sets it itself.

Tasks 1-5 are real Harbor task instructions (contextbench "cloud" suite, from
deepagents/libs/evals/datasets/context-retrieval-evals) -- ground truth for
those lives in that Harbor dataset. Tasks 6-9 are hand-authored, chosen to
need deliberately different tool subsets (execute-heavy, read-only, edit-heavy,
delegation-heavy) since the contextbench task family is tool-homogeneous
(every task needs read+search+write-answer) and doesn't exercise selection
differences. Tasks 6-9 have no fixture/environment or ground truth yet -- this
script only measures which tools get selected/called, not task correctness.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from unittest.mock import MagicMock

from deepagents.backends import LocalShellBackend
from deepagents.middleware import FilesystemMiddleware, SubAgentMiddleware
from langchain.agents.middleware.tool_selection import LLMToolSelectorMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from langchain_typesafe.experimental.middleware.tool_selector import (
    TsToolSelectorMiddleware,
)

MODEL = "gpt-5.6-luna"

TASKS = {
    # --- contextbench [harbor], homogeneous lookup+write-answer shape ---
    "cb-cloud-1": (
        "Among all people who live in the same state as the owner of the pet "
        "named 'Dawn', who owns the most vehicles? If there's a tie, who among "
        "them is the oldest?\n\nUse only the files under `/app/files`. Write "
        "your final answer (and nothing else) to `/app/answer.txt`."
    ),
    "cb-cloud-6": (
        "What is the total bank balance of the person who has the most credit "
        "cards among all residents of the same state as the owner of the pet "
        "named 'Betty'? If there's a tie for most credit cards, use the "
        "highest total bank balance as a tiebreaker.\n\nUse only the files "
        "under `/app/files`. Write your final answer (and nothing else) to "
        "`/app/answer.txt`."
    ),
    "cb-cloud-9": (
        "Among all people who live in the same state as the person with "
        "internet username 'reedjustin', who does NOT have any credit "
        "cards?\n\nUse only the files under `/app/files`. Write your final "
        "answer (and nothing else) to `/app/answer.txt`."
    ),
    "cb-cloud-22": (
        "Among all people who have an internet account at the same URL as the "
        "owner of the vehicle with license plate 'EUB7194', who does NOT own "
        "any vehicles?\n\nUse only the files under `/app/files`. Write your "
        "final answer (and nothing else) to `/app/answer.txt`."
    ),
    "cb-cloud-53": (
        "Among people who have exactly 6 bank accounts, who owns a vehicle "
        "with the same make as a vehicle owned by the owner of the pet named "
        "'Ana'?\n\nUse only the files under `/app/files`. Write your final "
        "answer (and nothing else) to `/app/answer.txt`."
    ),
    # --- mixed [authored], deliberately different tool requirements ---
    "run-tests": (
        "Run the test suite in this repo and report which tests are failing "
        "and why. Don't fix anything yet, just report."
    ),
    "explain-only": (
        "What does the function `parse_config` in config.py do? Just explain "
        "it in plain English -- don't change anything."
    ),
    "rename-var": (
        "Rename the variable `usr` to `user` everywhere it's used across the "
        "codebase."
    ),
    "delegate-refactor": (
        "This module needs a large refactor spanning many files. Delegate the "
        "whole thing to a subagent and have it handle it end to end."
    ),
}
REPS_PER_PATH = 3


def build_dcode_tools(model: object) -> list:
    scratch_dir = tempfile.mkdtemp(prefix="eval_tool_selector_")
    backend = LocalShellBackend(root_dir=scratch_dir, virtual_mode=False, inherit_env=False)
    fs_middleware = FilesystemMiddleware(
        backend=backend,
        tools=["ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"],
    )
    subagent_middleware = SubAgentMiddleware(
        backend=backend,
        subagents=[
            {
                "name": "general-purpose",
                "description": "General-purpose subagent for delegated work.",
                "model": model,
                "tools": [],
            }
        ],
    )
    return list(fs_middleware.tools) + list(subagent_middleware.tools)


def run_selection(middleware, model: object, query: str, tools: list) -> tuple[list, float]:
    request = ModelRequest(model=model, messages=[HumanMessage(query)], tools=list(tools))
    seen: list[ModelRequest] = []

    def handler(modified: ModelRequest) -> MagicMock:
        seen.append(modified)
        return MagicMock()

    start = time.perf_counter()
    middleware.wrap_model_call(request, handler)
    elapsed = time.perf_counter() - start
    filtered = [t for t in seen[0].tools if not isinstance(t, dict)]
    return filtered, elapsed


def run_downstream(model: ChatOpenAI, query: str, tools: list) -> tuple[list[str], float]:
    bound = model.bind_tools(tools)
    start = time.perf_counter()
    response = bound.invoke([HumanMessage(query)])
    elapsed = time.perf_counter() - start
    return [tc["name"] for tc in response.tool_calls], elapsed


@traceable(name="LLM tool selector run")
def run_llm_once(model: ChatOpenAI, query: str, tools: list) -> dict:
    middleware = LLMToolSelectorMiddleware(model=model)
    filtered, select_s = run_selection(middleware, model, query, tools)
    called, downstream_s = run_downstream(model, query, filtered)
    return {
        "selected": sorted(t.name for t in filtered),
        "called": called,
        "select_ms": select_s * 1000,
        "downstream_ms": downstream_s * 1000,
        "total_ms": (select_s + downstream_s) * 1000,
    }


@traceable(name="Jev tool selector run")
def run_jev_once(model: ChatOpenAI, query: str, tools: list) -> dict:
    middleware = TsToolSelectorMiddleware()  # relevance_threshold defaults to 0.5
    filtered, select_s = run_selection(middleware, model, query, tools)
    called, downstream_s = run_downstream(model, query, filtered)
    return {
        "selected": sorted(t.name for t in filtered),
        "called": called,
        "select_ms": select_s * 1000,
        "downstream_ms": downstream_s * 1000,
        "total_ms": (select_s + downstream_s) * 1000,
    }


@traceable(name="Tool selector eval (LLM vs Jev)")
def main() -> None:
    model = ChatOpenAI(model=MODEL, use_responses_api=True)
    tools = build_dcode_tools(model)
    print(f"Real dcode tool roster ({len(tools)} tools): {[t.name for t in tools]}")
    print(f"Tasks: {list(TASKS)}  ({REPS_PER_PATH} reps per path per task)\n")

    all_llm_ms: list[float] = []
    all_jev_ms: list[float] = []

    for task_name, query in TASKS.items():
        print(f"===== {task_name} =====")
        print(f"{query.splitlines()[0][:100]}...\n")

        for i in range(REPS_PER_PATH):
            r = run_llm_once(model, query, tools)
            all_llm_ms.append(r["total_ms"])
            print(
                f"  [LLM run {i + 1}] select={r['select_ms']:.0f}ms "
                f"downstream={r['downstream_ms']:.0f}ms total={r['total_ms']:.0f}ms "
                f"selected={r['selected']} called={r['called']}"
            )

        for i in range(REPS_PER_PATH):
            r = run_jev_once(model, query, tools)
            all_jev_ms.append(r["total_ms"])
            print(
                f"  [Jev run {i + 1}] select={r['select_ms']:.0f}ms "
                f"downstream={r['downstream_ms']:.0f}ms total={r['total_ms']:.0f}ms "
                f"selected={r['selected']} called={r['called']}"
            )
        print()

    print("===== Aggregate latency (total = select + downstream) =====")
    print(
        f"LLM: n={len(all_llm_ms)} mean={statistics.mean(all_llm_ms):.0f}ms "
        f"median={statistics.median(all_llm_ms):.0f}ms "
        f"min={min(all_llm_ms):.0f}ms max={max(all_llm_ms):.0f}ms "
        f"stdev={statistics.stdev(all_llm_ms):.0f}ms"
    )
    print(
        f"Jev: n={len(all_jev_ms)} mean={statistics.mean(all_jev_ms):.0f}ms "
        f"median={statistics.median(all_jev_ms):.0f}ms "
        f"min={min(all_jev_ms):.0f}ms max={max(all_jev_ms):.0f}ms "
        f"stdev={statistics.stdev(all_jev_ms):.0f}ms"
    )


if __name__ == "__main__":
    main()
