"""Benchmark token usage across different defense strategies.

Compares token consumption for:
1. No defense (baseline)
2. CheckToolStrategy only
3. ParseDataStrategy only  
4. CombinedStrategy (CheckTool + ParseData)

Uses claude-sonnet-4-5 for consistent measurement via usage_metadata.
"""

import pytest

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


def get_token_usage(response):
    """Extract token usage from response metadata."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        return {
            "input": response.usage_metadata.get("input_tokens", 0),
            "output": response.usage_metadata.get("output_tokens", 0),
            "total": response.usage_metadata.get("total_tokens", 0),
        }
    return {"input": 0, "output": 0, "total": 0}


class TokenTracker:
    """Track cumulative token usage across multiple calls."""
    
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
    
    def add(self, usage):
        self.input_tokens += usage.get("input", 0)
        self.output_tokens += usage.get("output", 0)
        self.calls += 1
    
    @property
    def total(self):
        return self.input_tokens + self.output_tokens
    
    def __repr__(self):
        return f"in={self.input_tokens} out={self.output_tokens} total={self.total} calls={self.calls}"


@pytest.fixture
def sonnet_model():
    """Create Claude Sonnet model for benchmarking."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model="claude-sonnet-4-5")


TEST_PAYLOADS = [
    ("goal_hijacking", GOAL_HIJACKING_PAYLOAD_1, [search_docs, send_email], "search_docs"),
    ("indirect_trigger", INDIRECT_TOOL_TRIGGER_PAYLOAD_1, [read_file, delete_file], "read_file"),
    ("json_injection", JSON_INJECTION_PAYLOAD_1, [send_email, get_user_data], "get_user_data"),
]


@pytest.mark.requires("langchain_anthropic")
class TestTokenBenchmark:
    """Benchmark token usage across defense strategies."""

    def test_token_usage_comparison(self, sonnet_model):
        """Compare token usage across all strategies for multiple payloads."""
        from langchain_core.messages import HumanMessage
        
        results = {
            "no_defense": TokenTracker(),
            "check_only": TokenTracker(),
            "parse_only": TokenTracker(),
            "combined": TokenTracker(),
        }
        
        for name, payload, tools, tool_name in TEST_PAYLOADS:
            # No defense - just invoke model with payload
            model_with_tools = sonnet_model.bind_tools(tools)
            response = model_with_tools.invoke([
                HumanMessage(content=f"Process this tool result:\n\n{payload}")
            ])
            results["no_defense"].add(get_token_usage(response))
            
            # CheckToolStrategy only
            check_strategy = CheckToolStrategy(sonnet_model, tools=tools)
            req = create_tool_request(tools, tool_name)
            msg = create_tool_message(payload, tool_name)
            
            # Wrap to capture token usage
            original_invoke = sonnet_model.invoke
            check_tokens = TokenTracker()
            def tracking_invoke(*args, **kwargs):
                resp = original_invoke(*args, **kwargs)
                check_tokens.add(get_token_usage(resp))
                return resp
            sonnet_model.invoke = tracking_invoke
            check_strategy.process(req, msg)
            sonnet_model.invoke = original_invoke
            results["check_only"].add({"input": check_tokens.input_tokens, "output": check_tokens.output_tokens})
            
            # ParseDataStrategy only
            parse_strategy = ParseDataStrategy(sonnet_model, use_full_conversation=True)
            parse_tokens = TokenTracker()
            def tracking_invoke2(*args, **kwargs):
                resp = original_invoke(*args, **kwargs)
                parse_tokens.add(get_token_usage(resp))
                return resp
            sonnet_model.invoke = tracking_invoke2
            parse_strategy.process(req, msg)
            sonnet_model.invoke = original_invoke
            results["parse_only"].add({"input": parse_tokens.input_tokens, "output": parse_tokens.output_tokens})
            
            # Combined strategy
            combined_strategy = CombinedStrategy([
                CheckToolStrategy(sonnet_model, tools=tools),
                ParseDataStrategy(sonnet_model, use_full_conversation=True),
            ])
            combined_tokens = TokenTracker()
            def tracking_invoke3(*args, **kwargs):
                resp = original_invoke(*args, **kwargs)
                combined_tokens.add(get_token_usage(resp))
                return resp
            sonnet_model.invoke = tracking_invoke3
            combined_strategy.process(req, msg)
            sonnet_model.invoke = original_invoke
            results["combined"].add({"input": combined_tokens.input_tokens, "output": combined_tokens.output_tokens})
        
        # Print results
        print("\n" + "="*70)
        print("TOKEN USAGE BENCHMARK (claude-sonnet-4-5)")
        print("="*70)
        print(f"Payloads tested: {len(TEST_PAYLOADS)}")
        print("-"*70)
        print(f"{'Strategy':<20} {'Input':>10} {'Output':>10} {'Total':>10} {'Calls':>8}")
        print("-"*70)
        
        for strategy, tracker in results.items():
            print(f"{strategy:<20} {tracker.input_tokens:>10} {tracker.output_tokens:>10} {tracker.total:>10} {tracker.calls:>8}")
        
        print("-"*70)
        
        # Calculate overhead vs no_defense
        baseline = results["no_defense"].total
        if baseline > 0:
            print("\nOverhead vs no_defense:")
            for strategy, tracker in results.items():
                if strategy != "no_defense":
                    overhead = ((tracker.total - baseline) / baseline) * 100
                    print(f"  {strategy}: {overhead:+.1f}%")
        
        print("="*70)
        
        # Assert combined uses more tokens than no defense (sanity check)
        assert results["combined"].total >= results["no_defense"].total, \
            "Combined strategy should use at least as many tokens as no defense"

    def test_per_payload_breakdown(self, sonnet_model):
        """Show token usage breakdown per payload type."""
        from langchain_core.messages import HumanMessage
        
        print("\n" + "="*70)
        print("PER-PAYLOAD TOKEN BREAKDOWN (combined strategy)")
        print("="*70)
        
        for name, payload, tools, tool_name in TEST_PAYLOADS:
            combined_strategy = CombinedStrategy([
                CheckToolStrategy(sonnet_model, tools=tools),
                ParseDataStrategy(sonnet_model, use_full_conversation=True),
            ])
            
            tracker = TokenTracker()
            original_invoke = sonnet_model.invoke
            def tracking_invoke(*args, **kwargs):
                resp = original_invoke(*args, **kwargs)
                tracker.add(get_token_usage(resp))
                return resp
            sonnet_model.invoke = tracking_invoke
            
            req = create_tool_request(tools, tool_name)
            msg = create_tool_message(payload, tool_name)
            combined_strategy.process(req, msg)
            
            sonnet_model.invoke = original_invoke
            
            print(f"{name:<25} input={tracker.input_tokens:>6} output={tracker.output_tokens:>6} total={tracker.total:>6} calls={tracker.calls}")
        
        print("="*70)
