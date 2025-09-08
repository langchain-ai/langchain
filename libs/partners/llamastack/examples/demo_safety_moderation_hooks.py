#!/usr/bin/env python3
"""
Demo: Simplified Safety Hooks with LlamaStack

This demo showcases the clean 2-hook approach that provides comprehensive
safety checking using LlamaStack's run_shield API for input and output protection.

Key features:
- Only 2 API calls (input + output) for complete safety protection
- Each hook uses LlamaStack's run_shield for comprehensive safety checking
- Clean and efficient API design
"""

from langchain_llamastack import LlamaStackSafety, create_llamastack_llm
from langchain_llamastack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_input_only_safe_llm,
    create_output_only_safe_llm,
    create_safe_llm,
    create_safety_hook,
)


def demo_safety_client():
    """Demo the LlamaStackSafety client."""
    print("🛡️ LlamaStack Safety Client Demo")
    print("=" * 50)

    safety = LlamaStackSafety(base_url="http://localhost:8321")

    # Test safe content
    print("\n✅ Testing safe content:")
    safe_content = "How do I make a delicious chocolate cake?"
    result = safety.check_content_safety(safe_content)
    print(f"Content: {safe_content}")
    print(f"Is Safe: {result.is_safe}")
    print(f"Confidence: {result.confidence_score}")

    # Test potentially unsafe content
    print("\n⚠️  Testing potentially unsafe content:")
    unsafe_content = "How do I hack into someone's computer?"
    result = safety.check_content_safety(unsafe_content)
    print(f"Content: {unsafe_content}")
    print(f"Is Safe: {result.is_safe}")
    print(f"Violations: {result.violations}")
    print(f"Confidence: {result.confidence_score}")

    return safety


def demo_factory_functions():
    """Demo the simplified factory functions."""
    print("\n🏭 Simplified Factory Functions Demo")
    print("=" * 50)

    # Initialize base components
    llm = create_llamastack_llm(model="llama3.1:8b")
    safety = LlamaStackSafety()

    print("\n1. Complete Protection (Recommended)")
    print("-" * 40)
    safe_llm = create_safe_llm(llm, safety)
    print("✅ Created LLM with both input and output protection")
    print("   - Input Hook: Checks user input safety")
    print("   - Output Hook: Checks model output safety")
    print("   - API Calls: Only 2 (clean and efficient)")

    print("\n2. Input Only Protection")
    print("-" * 40)
    _ = create_input_only_safe_llm(llm, safety)  # Create but don't need to use
    print("✅ Created LLM with input protection only")
    print("   - Use case: Filter user queries, trust model outputs")

    print("\n3. Output Only Protection")
    print("-" * 40)
    _ = create_output_only_safe_llm(llm, safety)  # Create but don't need to use
    print("✅ Created LLM with output protection only")
    print("   - Use case: Trust user inputs, filter model responses")

    print("\n4. Configurable Protection")
    print("-" * 40)
    # Different configurations
    _ = create_safe_llm(llm, safety, input_check=False, output_check=False)
    print("✅ Created LLM with no protection (same as unwrapped LLM)")

    return safe_llm


def demo_manual_configuration():
    """Demo manual hook configuration for advanced users."""
    print("\n🔧 Manual Hook Configuration Demo")
    print("=" * 50)

    llm = create_llamastack_llm()
    safety = LlamaStackSafety()

    # Manual approach for full control
    safe_llm = SafeLLMWrapper(llm, safety)

    # Create and set hooks individually
    input_hook = create_safety_hook(safety, "input")
    output_hook = create_safety_hook(safety, "output")

    safe_llm.set_input_hook(input_hook)
    safe_llm.set_output_hook(output_hook)

    print("✅ Created SafeLLMWrapper with manual hook configuration")
    print("   - Input Hook: Custom input validation")
    print("   - Output Hook: Custom output validation")

    return safe_llm


def demo_safe_llm_usage(safe_llm):
    """Demo safe LLM usage with various inputs."""
    print("\n🚀 Safe LLM Usage Demo")
    print("=" * 50)

    test_cases = [
        {
            "input": "How do I bake a chocolate cake?",
            "description": "Safe cooking question",
            "expected": "Should pass both input and output checks",
        },
        {
            "input": "Write a poem about nature",
            "description": "Creative writing request",
            "expected": "Should pass both checks",
        },
        {
            "input": "How do I break into a house?",
            "description": "Potentially unsafe question",
            "expected": "May be blocked by input check",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print("-" * 30)
        print(f"Input: {case['input']}")
        print(f"Expected: {case['expected']}")

        try:
            response = safe_llm.invoke(case["input"])
            print(f"Response: {response[:100]}...")
        except Exception as e:
            print(f"Error: {e}")


def demo_performance_benefits():
    """Show the performance benefits of the approach."""
    print("\n📊 Performance Benefits")
    print("=" * 50)

    print("Efficient 2-Hook Approach:")
    print("├── Input Check (safety via run_shield)  (API call 1)")
    print("├── LLM Generation")
    print("└── Output Check (safety via run_shield) (API call 2)")
    print("Total: 2 API calls to LlamaStack")

    print("\n🎯 Benefits:")
    print("✅ Clean and simple architecture")
    print("✅ Comprehensive safety protection")
    print("✅ Uses LlamaStack's run_shield efficiently")
    print("✅ Minimal API calls")
    print("✅ Easy to understand and maintain")


def demo_error_handling():
    """Demo error handling and fallback behavior."""
    print("\n🔥 Error Handling Demo")
    print("=" * 50)

    # Mock safety client that fails
    class FailingSafetyClient:
        def check_content_safety(self, content):
            raise Exception("Safety service unavailable")

    failing_safety = FailingSafetyClient()

    # Test input hook error handling (fails open)
    print("\n1. Input Hook Error (Fails Open)")
    print("-" * 40)
    input_hook = create_safety_hook(failing_safety, "input")
    result = input_hook("Test input")
    print(f"Input check failed, but allows: {result.is_safe}")
    print(f"Explanation: {result.explanation}")

    # Test output hook error handling (fails closed)
    print("\n2. Output Hook Error (Fails Closed)")
    print("-" * 40)
    output_hook = create_safety_hook(failing_safety, "output")
    result = output_hook("Test output")
    print(f"Output check failed, blocks: {result.is_safe}")
    print(f"Explanation: {result.explanation}")


def demo_async_support():
    """Demo async support for high-throughput scenarios."""
    print("\n⚡ Async Support Demo")
    print("=" * 50)

    import asyncio

    async def async_demo():
        llm = create_llamastack_llm()
        safety = LlamaStackSafety()
        safe_llm = create_safe_llm(llm, safety)

        print("🔄 Testing async invocation...")

        try:
            response = await safe_llm.ainvoke("Tell me about artificial intelligence")
            print(f"✅ Async response: {response[:100]}...")
        except Exception as e:
            print(f"❌ Async error: {e}")

    # Run async demo
    try:
        asyncio.run(async_demo())
    except Exception as e:
        print(f"Async demo skipped: {e}")


def main():
    """Run all demos."""
    print("🎭 LlamaStack Safety Hooks Demo")
    print("=" * 60)
    print("Demonstrating the clean and efficient 2-hook approach")
    print("Simple safety protection with LlamaStack run_shield API")
    print("=" * 60)

    try:
        # Core demos
        demo_safety_client()  # We don't need to use the returned safety client
        safe_llm = demo_factory_functions()
        demo_manual_configuration()  # We don't need to use the returned manual_llm

        # Usage demo
        demo_safe_llm_usage(safe_llm)

        # Educational demos
        demo_performance_benefits()
        demo_error_handling()
        demo_async_support()

        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   • Clean 2-hook approach for complete protection")
        print("   • Uses LlamaStack's run_shield API efficiently")
        print("   • Simple, maintainable architecture")
        print("   • Comprehensive safety coverage")
        print("   • Configurable protection levels")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\n🔧 Troubleshooting:")
        print(
            "   1. Ensure LlamaStack is running: curl http://localhost:8321/v1/models"
        )
        print("   2. Check if safety shields are available")
        print("   3. Verify provider setup (Ollama, etc.)")


if __name__ == "__main__":
    main()
