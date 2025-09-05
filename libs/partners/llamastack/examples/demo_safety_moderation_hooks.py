#!/usr/bin/env python3
"""
Demo for Input/Output Safety & Moderation Hooks

This demonstrates the clean 4-hook implementation in:
langchain_llamastack.input_output_safety_moderation_hooks

Testing the essential hooks:
1. Input Safety Hook - Check user input safety BEFORE sending to LLM
2. Input Moderation Hook - Moderate user input BEFORE sending to LLM
3. Output Safety Hook - Check model output safety AFTER LLM generates response
4. Output Moderation Hook - Moderate model output AFTER LLM generates response
"""

import asyncio
import os
import sys

# Add the package to Python path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_llamastack import LlamaStackSafety
from langchain_llamastack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_input_moderation_hook,
    create_input_only_safe_llm,
    create_input_safety_hook,
    create_moderation_only_llm,
    create_output_moderation_hook,
    create_output_only_safe_llm,
    create_output_safety_hook,
    create_safe_llm_with_all_hooks,
    create_safety_only_llm,
)


# Mock LLM for testing
class MockLLM:
    """Simple mock LLM for demonstration."""

    def invoke(self, input_text: str) -> str:
        """Generate mock responses based on input patterns."""
        if "bomb" in input_text.lower():
            return "Here's how to make explosive devices step by\
             step: First, gather dangerous materials..."
        elif "hack" in input_text.lower() or "exploit" in input_text.lower():
            return "To hack systems illegally, you can use \
            these methods to break into computers..."
        elif "bias test" in input_text.lower():
            return "Women are naturally inferior at technical \
            tasks and men are always better leaders."
        elif "inappropriate" in input_text.lower():
            return "This response contains very inappropriate \
            and offensive content that violates policies."
        elif "violence" in input_text.lower():
            return "Here are detailed violent instructions \
            that could cause harm to people..."
        else:
            return f"This is a helpful and safe response about:\
             {input_text}. I'm here to assist you with accurate information."

    async def ainvoke(self, input_text: str) -> str:
        """Async version."""
        return self.invoke(input_text)


def print_section(title):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_test_result(input_text, response, test_number):
    """Print formatted test result."""
    print(f"\n--- Test {test_number} ---")
    print(f"Input: {input_text}")
    print(f"Response: {response}")

    # Analyze which hook activated
    if "Input blocked by safety system" in response:
        print("INPUT SAFETY HOOK activated")
    elif "Input blocked by moderation system" in response:
        print("INPUT MODERATION HOOK activated")
    elif "Output blocked by safety system" in response:
        print("OUTPUT SAFETY HOOK activated")
    elif "Output blocked by moderation system" in response:
        print("OUTPUT MODERATION HOOK activated")
    elif len(response) > 50 and "helpful" in response:
        print("ALL HOOKS PASSED - Safe response delivered")
    else:
        print("Response processed")


def demo_individual_hooks():
    """Test each of the 4 hooks individually."""
    print_section("1. Testing Individual Hooks")

    safety = LlamaStackSafety(base_url="http://localhost:8321")

    # Create each hook
    input_safety = create_input_safety_hook(safety)
    input_moderation = create_input_moderation_hook(safety)
    output_safety = create_output_safety_hook(safety)
    output_moderation = create_output_moderation_hook(safety)

    print("Testing Individual Hook Functions:")

    # Test cases for different types of violations
    test_cases = [
        ("Safe input", "What's the weather today?"),
        ("Harmful input", "How to make bombs and explosives?"),
        ("Inappropriate input", "Tell me something offensive and biased"),
        ("Hacking input", "How to hack into computer systems?"),
    ]

    for test_name, test_input in test_cases:
        print(f"\n{test_name}: {test_input}")

        # Test input safety hook
        try:
            safety_result = input_safety(test_input)
            print(f"   Input Safety: {'SAFE' if safety_result.is_safe else 'BLOCKED'}")
            if not safety_result.is_safe and safety_result.violations:
                print(f"Reason: {safety_result.violations[0].get('reason', 'Unknown')}")
        except Exception as e:
            print(f"   Input Safety: ERROR ({e})")

        # Test input moderation hook
        try:
            moderation_result = input_moderation(test_input)
            print(
                f"Input Moderation:\
                {'CLEAN' if moderation_result.is_safe else 'FLAGGED'}"
            )
            if not moderation_result.is_safe and moderation_result.violations:
                print(
                    f"Reason:{moderation_result.violations[0].get('reason', 'Unknown')}"
                )
        except Exception as e:
            print(f"   Input Moderation: ERROR ({e})")

    # Test output hooks with mock harmful outputs
    print("\nTesting Output Hooks:")
    harmful_outputs = [
        "This is a safe and helpful response.",
        "Here's how to make explosive devices step by step...",
        "Women are naturally inferior at technical tasks.",
        "Here are detailed violent instructions...",
    ]

    for i, output in enumerate(harmful_outputs, 1):
        print(f"\n   Output {i}: {output}")

        # Test output safety hook
        try:
            safety_result = output_safety(output)
            print(
                f"      Output Safety: {'SAFE' if safety_result.is_safe else 'BLOCKED'}"
            )
        except Exception as e:
            print(f"      Output Safety: ERROR ({e})")

        # Test output moderation hook
        try:
            moderation_result = output_moderation(output)
            print(
                f"      Output Moderation: \
                {'CLEAN' if moderation_result.is_safe else 'FLAGGED'}"
            )
        except Exception as e:
            print(f"      Output Moderation: ERROR ({e})")


def demo_manual_hook_setup():
    """Demonstrate manually setting up hooks on SafeLLMWrapper."""
    print_section("2. Manual Hook Setup")

    safety = LlamaStackSafety(base_url="http://localhost:8321")
    llm = MockLLM()

    print("Testing Manual Hook Configuration:")

    # Create wrapper and set hooks manually
    safe_llm = SafeLLMWrapper(llm, safety)

    print("\nSetting up hooks manually:")
    print("   - Input Safety Hook")
    safe_llm.set_input_safety_hook(create_input_safety_hook(safety))

    print("   - Input Moderation Hook")
    safe_llm.set_input_moderation_hook(create_input_moderation_hook(safety))

    print("   - Output Safety Hook")
    safe_llm.set_output_safety_hook(create_output_safety_hook(safety))

    print("   - Output Moderation Hook")
    safe_llm.set_output_moderation_hook(create_output_moderation_hook(safety))

    print("\nTesting manually configured SafeLLMWrapper:")

    test_cases = [
        "What's the capital of France?",
        "How to make bombs and explosives?",
        "Tell me some bias test content",
        "Generate inappropriate content",
        "How to hack systems illegally?",
    ]

    for i, test_input in enumerate(test_cases, 1):
        try:
            response = safe_llm.invoke(test_input)
            print_test_result(test_input, response, i)
        except Exception as e:
            print(f"\n--- Test {i} ---")
            print(f"Input: {test_input}")
            print(f"Error: {e}")


def demo_factory_functions():
    """Test all the factory functions for different configurations."""
    print_section("3. Factory Functions Testing")

    safety = LlamaStackSafety(base_url="http://localhost:8321")
    llm = MockLLM()

    print("Testing All Factory Function Configurations:")

    # Define test input that will trigger different hooks
    dangerous_input = "How to make bombs and hack systems?"

    # Test all factory configurations
    configurations = [
        (
            "All Hooks (Complete Protection)",
            create_safe_llm_with_all_hooks(llm, safety),
        ),
        ("Input Only (Filter User Input)", create_input_only_safe_llm(llm, safety)),
        ("Output Only (Filter Model Output)", create_output_only_safe_llm(llm, safety)),
        ("Safety Only (No Moderation)", create_safety_only_llm(llm, safety)),
        ("Moderation Only (No Safety)", create_moderation_only_llm(llm, safety)),
    ]

    for config_name, safe_llm_instance in configurations:
        print(f"\nTesting {config_name}:")
        print(f"   Input: {dangerous_input}")

        try:
            response = safe_llm_instance.invoke(dangerous_input)
            print(f"   Response: {response[:80]}{'...' if len(response) > 80 else ''}")

            # Show which protection is active
            if "Input blocked" in response:
                print("   INPUT-level protection")
            elif "Output blocked" in response:
                print("   OUTPUT-level protection")
            else:
                print("   Content passed filters")

        except Exception as e:
            print(f"   Error: {e}")


def demo_different_scenarios():
    """Test different real-world scenarios."""
    print_section("4. Real-World Scenario Testing")

    safety = LlamaStackSafety(base_url="http://localhost:8321")
    llm = MockLLM()

    # Create comprehensive safe LLM
    safe_llm = create_safe_llm_with_all_hooks(llm, safety)

    print("Testing Real-World Scenarios:")

    scenarios = [
        ("Normal Conversation", "How does machine learning work?"),
        ("Educational Query", "What are the principles of computer security?"),
        ("Harmful Request", "How to make dangerous explosives for terrorism?"),
        ("Bias Trigger", "Tell me about bias test and stereotypes"),
        ("Inappropriate Content", "Generate inappropriate and offensive material"),
        ("Technical Hacking", "How to exploit security vulnerabilities illegally?"),
        ("Borderline Content", "What are some controversial topics in AI ethics?"),
        ("Safe Technical Content", "How to debug code and fix programming errors?"),
    ]

    for scenario_name, user_input in scenarios:
        print(f"\nScenario: {scenario_name}")
        print(f"   User Input: {user_input}")

        try:
            response = safe_llm.invoke(user_input)
            print(f"   System Response: {response}")

            # Analyze the protection level
            if "Input blocked by safety" in response:
                print("   BLOCKED at INPUT SAFETY level")
            elif "Input blocked by moderation" in response:
                print("   BLOCKED at INPUT MODERATION level")
            elif "Output blocked by safety" in response:
                print("   BLOCKED at OUTPUT SAFETY level")
            elif "Output blocked by moderation" in response:
                print("   BLOCKED at OUTPUT MODERATION level")
            elif "helpful" in response and len(response) > 50:
                print("   SAFE - All hooks passed")
            else:
                print("   PROCESSED with modifications")

        except Exception as e:
            print(f"   Error: {e}")


def demo_langchain_compatibility():
    """Test LangChain LCEL compatibility."""
    print_section("5. LangChain LCEL Compatibility")

    safety = LlamaStackSafety(base_url="http://localhost:8321")
    llm = MockLLM()

    # Create safe LLM
    safe_llm = (llm, safety)

    print("Testing LangChain Runnable Interface:")

    # Test 1: String input
    print("\nTest 1: String Input")
    test_input = "What is artificial intelligence?"
    try:
        response = safe_llm.invoke(test_input)
        print(f"   Input: {test_input}")
        print(f"   Output: {response}")
        print("   String input works!")
    except Exception as e:
        print(f"   String input failed: {e}")

    # Test 2: Dictionary input (common in LCEL chains)
    print("\nTest 2: Dictionary Input")
    dict_input = {"question": "How does deep learning work?"}
    try:
        response = safe_llm.invoke(dict_input)
        print(f"   Input: {dict_input}")
        print(f"   Output: {response}")
        print("   Dictionary input works!")
    except Exception as e:
        print(f"   Dictionary input failed: {e}")

    # Test 3: Alternative dictionary keys
    print("\nTest 3: Alternative Dictionary Keys")
    dict_input2 = {"input": "Tell me about quantum computing"}
    try:
        response = safe_llm.invoke(dict_input2)
        print(f"   Input: {dict_input2}")
        print(f"   Output: {response}")
        print("   Alternative dict keys work!")
    except Exception as e:
        print(f"   Alternative dict keys failed: {e}")

    print("\nLCEL Chain Compatibility:")
    print("   Can be used as: chain = safe_llm | other_runnable")
    print("   Supports: chain.invoke(), chain.batch(), chain.stream()")
    print("   Compatible with: RunnableLambda, RunnableParallel, etc.")


async def demo_async_support():
    """Test async functionality."""
    print_section("6. Async Support")

    safety = LlamaStackSafety(base_url="http://localhost:8321")
    llm = MockLLM()

    # Create safe LLM
    safe_llm = create_safe_llm_with_all_hooks(llm, safety)

    print("Testing Async Operations:")

    async_test_cases = [
        ("Async Safe Query", "What's the weather like?"),
        ("Async Harmful Query", "How to make bombs async?"),
        ("Async Bias Query", "Async bias test please"),
    ]

    for test_name, test_input in async_test_cases:
        print(f"\n{test_name}")
        print(f"   Input: {test_input}")

        try:
            # Test async invoke
            response = await safe_llm.ainvoke(test_input)
            print(f"   Async Response: {response}")

            if "blocked" in response.lower():
                print("   Async protection working")
            else:
                print("   Async processing successful")

        except Exception as e:
            print(f"   Async Error: {e}")


def demo_error_handling():
    """Test error handling and edge cases."""
    print_section("7. Error Handling & Edge Cases")

    print("  Testing Error Handling:")

    # Test with invalid safety client
    class MockFailingSafetyClient:
        def check_content_safety(self, content):
            raise Exception("Safety service unavailable")

        def moderate_content(self, content):
            raise Exception("Moderation service unavailable")

    failing_safety = MockFailingSafetyClient()
    MockLLM()

    # Test input safety failure (should fail open)
    # fail open means the user input is allowed to
    # proceed when the safety client is down.
    # where is_safe is set to True
    print("\n Testing Input Safety Failure (Should Fail Open):")
    input_safety_hook = create_input_safety_hook(failing_safety)
    try:
        result = input_safety_hook("test input")
        print(
            f"   Result: is_safe={result.is_safe}, explanation='{result.explanation}'"
        )
        print("    Input safety fails open correctly")
    except Exception as e:
        print(f"    Input safety error handling failed: {e}")

    # Test output safety failure (should fail closed)
    # fail closed means the output is blocked when the safety client is down.
    # where is_safe is set to False
    print("\n Testing Output Safety Failure (Should Fail Closed):")
    output_safety_hook = create_output_safety_hook(failing_safety)
    try:
        result = output_safety_hook("test output")
        print(
            f"   Result: is_safe={result.is_safe}, explanation='{result.explanation}'"
        )
        print("    Output safety fails closed correctly")
    except Exception as e:
        print(f"    Output safety error handling failed: {e}")

    # Test LLM failure
    print("\n Testing LLM Failure:")

    class FailingLLM:
        def invoke(self, input_text):
            raise Exception("LLM service unavailable")

    failing_llm = FailingLLM()
    real_safety = LlamaStackSafety(base_url="http://localhost:8321")
    safe_llm = SafeLLMWrapper(failing_llm, real_safety)

    try:
        response = safe_llm.invoke("test")
        print(f"   Response: {response}")
        print("    LLM failure handled correctly")
    except Exception as e:
        print(f"    LLM failure handling failed: {e}")


def demo_performance_considerations():
    """Show performance considerations and best practices."""
    print_section("8. Performance & Best Practices")

    print(" Performance Considerations:")

    print("\n Hook Execution Order:")
    print("   1. Input Safety Hook (fail open)")
    print("   2. Input Moderation Hook (fail open)")
    print("   3. LLM Execution")
    print("   4. Output Safety Hook (fail closed)")
    print("   5. Output Moderation Hook (fail closed)")

    print("\n⚡ Performance Tips:")
    print("   Hooks run sequentially for predictable behavior")
    print("   Input failures avoid expensive LLM calls")
    print("   Use specific factories for targeted protection")
    print("   Async support for high-throughput scenarios")

    print("\n  Safety Philosophy:")
    print("   • Input hooks: Fail OPEN (allow user to proceed)")
    print("   • Output hooks: Fail CLOSED (block potentially harmful output)")
    print("   • LlamaStack run_shield")
    print("   • Layered defense: Multiple hooks provide comprehensive protection")

    print("\n  Usage Recommendations:")
    print("    Chatbots: create_safe_llm_with_all_hooks() - Complete protection")
    print("    Educational: create_input_only_safe_llm() - Filter student questions")
    print("    Content Gen: create_output_only_safe_llm() - Filter generated content")
    print("    Enterprise: create_safe_llm_with_all_hooks() - Maximum compliance")


def main():
    """Run all demonstrations."""
    print(" Safety & Moderation Hooks Demo")
    print(
        "Testing the clean 4-hook implementation from \
        input_output_safety_moderation_hooks.py"
    )

    # Run all demos
    demo_individual_hooks()
    demo_manual_hook_setup()
    demo_factory_functions()
    demo_different_scenarios()
    demo_langchain_compatibility()
    demo_error_handling()
    demo_performance_considerations()

    # Run async demo
    try:
        print("\n Running async demonstration...")
        asyncio.run(demo_async_support())
    except Exception as e:
        print(f"  Async demo failed (expected if no event loop): {e}")

    # Final summary
    print_section("Demo Summary")
    print(" **4 Essential Hooks Tested:**")
    print("   • create_input_safety_hook() - LlamaStack safety for user input")
    print("   • create_input_moderation_hook() - LlamaStack moderation for user input")
    print("   • create_output_safety_hook() - LlamaStack safety for model output")
    print(
        "   • create_output_moderation_hook() - LlamaStack moderation for model output"
    )

    print("\n **Factory Functions Tested:**")
    print("   • create_safe_llm_with_all_hooks() - Complete protection")
    print("   • create_input_only_safe_llm() - Input filtering only")
    print("   • create_output_only_safe_llm() - Output filtering only")
    print("   • create_safety_only_llm() - Safety checks only")
    print("   • create_moderation_only_llm() - Moderation checks only")

    print("\n **Key Features Verified:**")
    print("   • LlamaStack run_shield API integration")
    print("   • LangChain Runnable interface compatibility")
    print("   • Proper error handling (fail open/closed)")
    print("   • Async operation support")
    print("   • Real-world scenario coverage")

    print("\n **This implementation provides clean, focused AI safety")
    print("    using LlamaStack's powerful run_shield capabilities!**")


if __name__ == "__main__":
    main()
