#!/usr/bin/env python3
"""
Getting Started Guide for LangChain LlamaStack
==============================================

Quick setup and usage examples for the LangChain LlamaStack package.
"""

# =============================================================================
# QUICK SETUP GUIDE
# =============================================================================

setup_guide = """
# Quick Setup
=============

1. Install: pip install -e /path/to/langchain_llamastack_integration/langchain/libs/partners/llamastack

2. Set provider environment variables (optional):
   export FIREWORKS_API_KEY="your-key"
   export TOGETHER_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"

3. Start LlamaStack:
   export OLLAMA_URL=http://localhost:11434
   uv run --with llama-stack==0.2.18 llama stack build --distro starter --image-type venv --run

4. Verify: curl http://localhost:8321/v1/models
"""

print(setup_guide)


def check_setup():
    """Check if LlamaStack is accessible and has models."""
    print("🔍 Checking Setup...")
    try:
        from langchain_llamastack import check_llamastack_status

        status = check_llamastack_status()
        if status["connected"]:
            print(f"✅ Connected! {status['models_count']} models available")
            return True
        else:
            print(f"❌ Connection failed: {status['error']}")
            return False
    except Exception as e:
        print(f"❌ Setup check failed: {e}")
        return False


def example_basic_chat():
    """Example 1: Basic chat completion."""
    print("\n🤖 Example 1: Basic Chat")
    print("-" * 30)

    try:
        from langchain_llamastack import create_llamastack_llm

        # Auto-select first available model
        llm = create_llamastack_llm()
        response = llm.invoke("What is AI in one sentence?")
        print(f"Response: {response.content}")
        print("✅ Chat example completed")

    except Exception as e:
        print(f"❌ Chat failed: {e}")


def example_specific_model():
    """Example 2: Use specific model with parameters."""
    print("\n🎯 Example 2: Specific Model")
    print("-" * 30)

    try:
        from langchain_llamastack import create_llamastack_llm

        # Use specific model with parameters
        llm = create_llamastack_llm(
            model="ollama/llama3:70b-instruct",
            temperature=0.7,
            auto_fallback=True,  # Falls back if model not available
        )
        response = llm.invoke("Tell me about machine learning briefly")
        print(f"Response: {response.content[:100]}...")
        print("✅ Specific model example completed")

    except Exception as e:
        print(f"❌ Specific model failed: {e}")


def example_streaming():
    """Example 3: Streaming responses."""
    print("\n🔄 Example 3: Streaming")
    print("-" * 30)

    try:
        from langchain_llamastack import create_llamastack_llm

        llm = create_llamastack_llm()
        print("Streaming response: ", end="", flush=True)

        for chunk in llm.stream("Count from 1 to 5"):
            print(chunk.content, end="", flush=True)

        print("\n✅ Streaming example completed")

    except Exception as e:
        print(f"❌ Streaming failed: {e}")


def example_embeddings():
    """Example 4: Text embeddings."""
    print("\n📊 Example 4: Embeddings")
    print("-" * 30)

    try:
        from langchain_llamastack import LlamaStackEmbeddings

        embeddings = LlamaStackEmbeddings(model="nomic-embed-text")

        # Single embedding
        text = "Hello world"
        embedding = embeddings.embed_query(text)
        print(f"Text: '{text}' -> Embedding dimension: {len(embedding)}")

        # Multiple documents
        docs = ["AI is powerful", "Python is easy", "LangChain is useful"]
        doc_embeddings = embeddings.embed_documents(docs)
        print(f"Generated {len(doc_embeddings)} document embeddings")
        print("✅ Embeddings example completed")

    except Exception as e:
        print(f"❌ Embeddings failed: {e}")


def example_safety():
    """Example 5: Safety checking."""
    print("\n🛡️ Example 5: Safety")
    print("-" * 30)

    try:
        from langchain_llamastack import LlamaStackSafety

        safety = LlamaStackSafety()
        shields = safety.get_available_shields()

        if not shields:
            print("⚠️ No shield models available")
            print("Install with: ollama pull shieldgemma:2b")
            return

        print(f"Available shields: {len(shields)}")

        # Test content
        test_content = "Hello, how are you?"
        result = safety.check_content(test_content)
        print(f"Content: '{test_content}' -> Safe: {result.is_safe}")
        print("✅ Safety example completed")

    except Exception as e:
        print(f"❌ Safety failed: {e}")


def example_multi_turn():
    """Example 6: Multi-turn conversation."""
    print("\n💬 Example 6: Multi-turn Chat")
    print("-" * 30)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_llamastack import create_llamastack_llm

        llm = create_llamastack_llm()

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's your name?"),
        ]

        response = llm.invoke(messages)
        print(f"Assistant: {response.content[:100]}...")
        print("✅ Multi-turn example completed")

    except Exception as e:
        print(f"❌ Multi-turn failed: {e}")


def show_quick_templates():
    """Show copy-paste templates."""
    print("\n📋 Quick Start Templates")
    print("=" * 40)

    templates = {
        "Basic Chat": """
from langchain_llamastack import create_llamastack_llm
llm = create_llamastack_llm()
response = llm.invoke("Hello!")
print(response.content)
""",
        "Embeddings": """
from langchain_llamastack import LlamaStackEmbeddings
embeddings = LlamaStackEmbeddings(model="nomic-embed-text")
embedding = embeddings.embed_query("Hello world")
print(f"Dimension: {len(embedding)}")
""",
        "Safety": """
from langchain_llamastack import LlamaStackSafety
safety = LlamaStackSafety()
result = safety.check_content("Hello world")
print(f"Safe: {result.is_safe}")
""",
    }

    for name, code in templates.items():
        print(f"\n{name}:")
        print(code.strip())


def troubleshooting():
    """Show common issues and solutions."""
    print("\n🔧 Troubleshooting")
    print("=" * 40)

    issues = """
Common Issues:

1. Connection Error:
   - Check: curl http://localhost:8321/v1/models
   - Start LlamaStack server if needed

2. No Models Found:
   - Install models: ollama pull llama3.1:8b
   - Check configuration

3. Safety Models Missing:
   - Install: ollama pull shieldgemma:2b
   - Restart LlamaStack

4. Import Error:
   - Install package properly
   - Check Python path
"""

    print(issues)


def main():
    """Run getting started examples."""
    print("🚀 LangChain LlamaStack - Getting Started")
    print("=" * 50)

    # Check setup first
    if not check_setup():
        print("\n💡 Setup required:")
        print("1. Start LlamaStack server")
        print("2. Install models: ollama pull llama3.1:8b")
        print("3. Restart this script")
        return

    # Run examples
    examples = [
        example_basic_chat,
        example_specific_model,
        example_streaming,
        example_embeddings,
        example_safety,
        example_multi_turn,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example failed: {e}")

    # Show templates and troubleshooting
    show_quick_templates()
    troubleshooting()

    print("\n🎉 Getting Started Complete!")
    print("Copy any template above to start building your application.")


if __name__ == "__main__":
    main()
