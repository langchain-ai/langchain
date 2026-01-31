from langchain_classic.callbacks.cost_tracking import CostTrackingCallback
from langchain_core.outputs import LLMResult

def test_cost_tracking_basic() -> None:
    # Setup
    cb = CostTrackingCallback(cost_per_1k_tokens=0.01)

    # Simulate manual update (or assume we want to test logic)
    # Since on_llm_end logic depends on LLMResult, let's test that logic
    # or just test the summary calculation as requested in the task description "TEST (Very Important)" section
    # The prompt explicitly asked for this specific test structure:

    cb.total_tokens = 2000
    cb.total_cost = 0.02

    summary = cb.summary()
    assert summary["total_tokens"] == 2000
    assert summary["total_cost_usd"] == 0.02

def test_cost_tracking_logic() -> None:
    cb = CostTrackingCallback(cost_per_1k_tokens=0.002)

    # Mock LLMResult
    llm_output = {"token_usage": {"total_tokens": 1000}}
    response = LLMResult(generations=[], llm_output=llm_output)

    cb.on_llm_start({}, [])
    cb.on_llm_end(response)

    summary = cb.summary()
    assert summary["total_tokens"] == 1000
    # 1000 tokens * 0.002 / 1000 = 0.002
    assert summary["total_cost_usd"] == 0.002
    assert summary["latency_sec"] >= 0
