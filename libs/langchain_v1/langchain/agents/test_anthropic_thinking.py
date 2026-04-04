def test_anthropic_thinking_bypass_any():
    # Mock an Anthropic model with thinking enabled
    model = ChatAnthropic(
        model_name="claude-3-7-sonnet-20250219",
        thinking={"type": "enabled", "budget_tokens": 1024}
    )
    # Trigger the agent factory logic
    # ASSERT that tool_choice is NOT "any"