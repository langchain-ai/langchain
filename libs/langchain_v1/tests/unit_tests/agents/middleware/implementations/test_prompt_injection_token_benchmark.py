"""Benchmark token usage across different defense strategies.

Compares token consumption for:
1. No defense (baseline)
2. CheckToolStrategy only
3. ParseDataStrategy only
4. CombinedStrategy (CheckTool + ParseData)

Uses claude-sonnet-4-5 for consistent measurement via usage_metadata.

NOTE: These tests are skipped by default in CI because they:
1. Make real API calls to LLM providers (costs money)
2. Are slow (multiple LLM roundtrips per test)
3. Are for manual benchmarking, not regression testing

To run these tests manually:
    pytest tests/unit_tests/agents/middleware/implementations/test_prompt_injection_token_benchmark.py -m "not skip_in_ci" --run-benchmark

Or set the environment variable:
    RUN_BENCHMARK_TESTS=1 pytest ...
"""

import os
from typing import Any

import pytest

# Skip all tests in this module by default unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark tests are skipped by default. Set RUN_BENCHMARK_TESTS=1 to run.",
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult

from langchain.agents.middleware import (
    CheckToolStrategy,
    CombinedStrategy,
    ParseDataStrategy,
)

from .conftest import (
    GOAL_HIJACKING_PAYLOAD_1,
    INDIRECT_TOOL_TRIGGER_PAYLOAD_1,
    JSON_INJECTION_PAYLOAD_1,
    create_tool_message,
    create_tool_request,
    delete_file,
    get_user_data,
    read_file,
    search_docs,
    send_email,
)


class TokenCountingCallback(BaseCallbackHandler):
    """Callback handler that tracks token usage."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # Try llm_output.usage first (Anthropic style)
        if response.llm_output and "usage" in response.llm_output:
            usage = response.llm_output["usage"]
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.calls += 1
        # Try generations[0].message.usage_metadata (Ollama/OpenAI style)
        elif response.generations and response.generations[0]:
            gen = (
                response.generations[0][0]
                if isinstance(response.generations[0], list)
                else response.generations[0]
            )
            if (
                hasattr(gen, "message")
                and hasattr(gen.message, "usage_metadata")
                and gen.message.usage_metadata
            ):
                usage = gen.message.usage_metadata
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.calls += 1

    @property
    def total(self):
        return self.input_tokens + self.output_tokens

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0


MALICIOUS_PAYLOADS = [
    ("goal_hijacking", GOAL_HIJACKING_PAYLOAD_1, [search_docs, send_email], "search_docs"),
    ("indirect_trigger", INDIRECT_TOOL_TRIGGER_PAYLOAD_1, [read_file, delete_file], "read_file"),
    ("json_injection", JSON_INJECTION_PAYLOAD_1, [send_email, get_user_data], "get_user_data"),
]

# Benign payloads - normal tool results without any injection attempts
BENIGN_PAYLOADS = [
    (
        "search_results",
        """Found 3 documents matching "quarterly report":

1. Q4 2024 Financial Summary.pdf (Updated: Jan 15, 2025)
2. Q3 2024 Revenue Analysis.xlsx (Updated: Oct 3, 2024)  
3. Annual Report 2024 Draft.docx (Updated: Jan 10, 2025)

Would you like me to open any of these files?""",
        [search_docs, send_email],
        "search_docs",
    ),
    (
        "email_search",
        """Found 5 emails matching your search:

From: alice@company.com - "Re: Project Timeline" - Jan 14
From: bob@company.com - "Meeting Notes" - Jan 13
From: carol@company.com - "Q4 Review" - Jan 12
From: dave@company.com - "Budget Approval" - Jan 11
From: eve@company.com - "Team Lunch Friday" - Jan 10

Showing 5 of 12 results.""",
        [search_docs, send_email],
        "search_email",
    ),
    (
        "file_contents",
        """File: /home/user/config.yaml

database:
  host: localhost
  port: 5432
  name: myapp_production
  
server:
  port: 8080
  workers: 4
  timeout: 30

logging:
  level: INFO
  format: json""",
        [read_file, delete_file],
        "read_file",
    ),
    (
        "api_response",
        """{
  "status": "success",
  "data": {
    "user_id": 12345,
    "name": "John Smith",
    "email": "john.smith@example.com",
    "role": "developer",
    "department": "Engineering",
    "start_date": "2023-06-15"
  },
  "timestamp": "2025-01-15T10:30:00Z"
}""",
        [send_email, get_user_data],
        "get_user_data",
    ),
    (
        "weather_data",
        """Current weather for San Francisco, CA:

Temperature: 58°F (14°C)
Conditions: Partly Cloudy
Humidity: 72%
Wind: 12 mph W
UV Index: 3 (Moderate)

Forecast:
- Today: High 62°F, Low 51°F, Partly cloudy
- Tomorrow: High 59°F, Low 49°F, Rain likely
- Wednesday: High 55°F, Low 47°F, Showers""",
        [search_docs, send_email],
        "web_search",
    ),
    (
        "code_output",
        """Execution completed successfully.

Output:
>>> print(sum([1, 2, 3, 4, 5]))
15
>>> print("Hello, World!")
Hello, World!
>>> len([x for x in range(100) if x % 2 == 0])
50

Runtime: 0.003s
Memory used: 2.1 MB""",
        [search_docs, send_email],
        "execute_code",
    ),
    (
        "calendar_events",
        """Upcoming events for this week:

Monday, Jan 20:
  9:00 AM - Team Standup (30 min)
  2:00 PM - Project Review with Sarah (1 hr)

Tuesday, Jan 21:
  10:00 AM - Client Call - Acme Corp (45 min)
  3:30 PM - 1:1 with Manager (30 min)

Wednesday, Jan 22:
  11:00 AM - Lunch with Marketing Team
  4:00 PM - Sprint Planning (2 hrs)""",
        [search_docs, send_email],
        "search_docs",
    ),
    (
        "translation_result",
        """Translation (English → Spanish):

Original: "The quick brown fox jumps over the lazy dog."
Translated: "El rápido zorro marrón salta sobre el perro perezoso."

Original: "Good morning, how can I help you today?"
Translated: "Buenos días, ¿cómo puedo ayudarle hoy?"

Confidence: 98%""",
        [send_email, get_user_data],
        "translate",
    ),
    (
        "database_query",
        """Query results (5 rows):

| id  | product_name    | price  | stock |
|-----|-----------------|--------|-------|
| 101 | Wireless Mouse  | $29.99 |   145 |
| 102 | USB Keyboard    | $49.99 |    87 |
| 103 | Monitor Stand   | $39.99 |    62 |
| 104 | Webcam HD       | $79.99 |    34 |
| 105 | USB Hub         | $24.99 |   203 |

Total rows: 5 | Query time: 12ms""",
        [search_docs, send_email],
        "search_docs",
    ),
    (
        "simple_response",
        """The answer to your question is: 42.

This is based on the calculation you requested:
- Base value: 10
- Multiplier: 4
- Addition: 2
- Result: (10 × 4) + 2 = 42

Let me know if you need anything else!""",
        [search_docs, send_email],
        "search_docs",
    ),
]


def run_benchmark(model, payloads, label, callback):
    """Run benchmark for a set of payloads and return results."""
    results = {}

    # No defense
    if callback:
        callback.reset()
    for name, payload, tools, tool_name in payloads:
        model_with_tools = model.bind_tools(tools)
        model_with_tools.invoke([HumanMessage(content=f"Process this tool result:\n\n{payload}")])
    results["no_defense"] = (
        {"input": callback.input_tokens, "output": callback.output_tokens, "calls": callback.calls}
        if callback
        else {}
    )

    # CheckToolStrategy only
    if callback:
        callback.reset()
    for name, payload, tools, tool_name in payloads:
        strategy = CheckToolStrategy(model, tools=tools)
        req = create_tool_request(tools, tool_name)
        strategy.process(req, create_tool_message(payload, tool_name))
    results["check_only"] = (
        {"input": callback.input_tokens, "output": callback.output_tokens, "calls": callback.calls}
        if callback
        else {}
    )

    # ParseDataStrategy only
    if callback:
        callback.reset()
    for name, payload, tools, tool_name in payloads:
        strategy = ParseDataStrategy(model, use_full_conversation=True)
        req = create_tool_request(tools, tool_name)
        strategy.process(req, create_tool_message(payload, tool_name))
    results["parse_only"] = (
        {"input": callback.input_tokens, "output": callback.output_tokens, "calls": callback.calls}
        if callback
        else {}
    )

    # Combined strategy
    if callback:
        callback.reset()
    for name, payload, tools, tool_name in payloads:
        strategy = CombinedStrategy(
            [
                CheckToolStrategy(model, tools=tools),
                ParseDataStrategy(model, use_full_conversation=True),
            ]
        )
        req = create_tool_request(tools, tool_name)
        strategy.process(req, create_tool_message(payload, tool_name))
    results["combined"] = (
        {"input": callback.input_tokens, "output": callback.output_tokens, "calls": callback.calls}
        if callback
        else {}
    )

    return results


def print_comparison_summary(model_name: str, malicious_results: dict, benign_results: dict):
    """Print comparison summary for malicious vs benign payloads."""
    print(f"\n{'=' * 80}")
    print(f"COMPARISON SUMMARY ({model_name})")
    print(f"{'=' * 80}")

    for label, results in [("MALICIOUS", malicious_results), ("BENIGN", benign_results)]:
        print(f"\n{label} payloads:")
        print(
            f"{'Strategy':<12} {'Input':>8} {'Output':>8} {'Total':>8} {'In Δ':>8} {'Out Δ':>8} {'Tot Δ':>8}"
        )
        print(f"{'-' * 80}")

        base_in = results["no_defense"]["input"]
        base_out = results["no_defense"]["output"]
        base_total = base_in + base_out

        for strategy in ["no_defense", "check_only", "parse_only", "combined"]:
            inp = results[strategy]["input"]
            out = results[strategy]["output"]
            total = inp + out

            if strategy == "no_defense":
                print(f"{strategy:<12} {inp:>8} {out:>8} {total:>8} {'--':>8} {'--':>8} {'--':>8}")
            else:
                in_oh = ((inp - base_in) / base_in * 100) if base_in > 0 else 0
                out_oh = ((out - base_out) / base_out * 100) if base_out > 0 else 0
                tot_oh = ((total - base_total) / base_total * 100) if base_total > 0 else 0
                print(
                    f"{strategy:<12} {inp:>8} {out:>8} {total:>8} {in_oh:>+7.1f}% {out_oh:>+7.1f}% {tot_oh:>+7.1f}%"
                )

    print(f"\n{'=' * 80}")


def _run_token_benchmark(model, model_name: str, callback: TokenCountingCallback):
    """Run token benchmark for a model and print results."""
    malicious_results = run_benchmark(model, MALICIOUS_PAYLOADS, "malicious", callback)
    benign_results = run_benchmark(model, BENIGN_PAYLOADS, "benign", callback)
    print_comparison_summary(model_name, malicious_results, benign_results)


@pytest.mark.requires("langchain_anthropic")
class TestTokenBenchmarkAnthropic:
    """Benchmark token usage for Anthropic (claude-opus-4-5)."""

    def test_comparison_summary(self):
        from langchain_anthropic import ChatAnthropic

        callback = TokenCountingCallback()
        model = ChatAnthropic(model="claude-opus-4-5", callbacks=[callback])
        _run_token_benchmark(model, "claude-opus-4-5", callback)


@pytest.mark.requires("langchain_openai")
class TestTokenBenchmarkOpenAI:
    """Benchmark token usage for OpenAI (gpt-5.2)."""

    def test_comparison_summary(self):
        from langchain_openai import ChatOpenAI

        callback = TokenCountingCallback()
        model = ChatOpenAI(model="gpt-5.2", callbacks=[callback])
        _run_token_benchmark(model, "gpt-5.2", callback)


@pytest.mark.requires("langchain_google_genai")
class TestTokenBenchmarkGoogle:
    """Benchmark token usage for Google (gemini-3-flash-preview)."""

    def test_comparison_summary(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        callback = TokenCountingCallback()
        model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", callbacks=[callback])
        _run_token_benchmark(model, "gemini-3-flash-preview", callback)


@pytest.mark.requires("langchain_ollama")
class TestTokenBenchmarkOllama:
    """Benchmark token usage for Ollama."""

    def test_comparison_summary(self):
        from langchain_ollama import ChatOllama

        from .conftest import OLLAMA_BASE_URL, OLLAMA_MODELS

        model_name = OLLAMA_MODELS[0]
        callback = TokenCountingCallback()
        model = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, callbacks=[callback])
        _run_token_benchmark(model, model_name, callback)
