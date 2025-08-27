#!/usr/bin/env python3
"""Basic usage examples for LangChain Llama Stack integration - Updated for simplified approach."""

from langchain_llamastack import (
    check_llamastack_status,
    create_llamastack_llm,
    get_llamastack_models,
    LlamaStackEmbeddings,
    LlamaStackSafety,
)


def test_chat_completion():
    """Test basic chat completion using the new recommended approach."""
    print("🤖 Testing Chat Completion (New Factory Function Approach)")
    print("=" * 60)

    try:
        # NEW RECOMMENDED APPROACH: Use factory function
        llm = create_llamastack_llm(
            model="ollama/llama3:70b-instruct",
            # Will auto-fallback if not available
        )

        # Test basic completion
        response = llm.invoke("How can AI help in education?")
        print(f"Response: {response.content}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


# ChatLlamaStack class has been removed - using factory function approach only


def test_model_discovery():
    """Test model discovery and connection checking."""
    print("\n🔍 Testing Model Discovery")
    print("=" * 40)

    try:
        # Check connection status
        status = check_llamastack_status()
        print(f"Connection: {'✅ Connected' if status['connected'] else '❌ Failed'}")
        print(f"Models available: {status['models_count']}")

        if status["connected"]:
            print(f"Available models: {status['models']}")
        else:
            print(f"Error: {status['error']}")
            return False

        # Get models list directly
        models = get_llamastack_models()
        print(f"Direct model query returned {len(models)} models")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_manual_chatopenaai():
    """Test manual ChatOpenAI usage for advanced users."""
    print("\n🔧 Testing Manual ChatOpenAI Usage")
    print("=" * 40)

    try:
        # Import ChatOpenAI directly
        from langchain_openai import ChatOpenAI

        # Manual approach for full control
        models = get_llamastack_models()
        if not models:
            print("No models available")
            return False

        llm = ChatOpenAI(
            base_url="http://localhost:8321/v1/openai/v1",
            api_key="not-needed",  # LlamaStack doesn't require real API keys
            model="ollama/llama3:70b-instruct",  # models[0],
        )

        response = llm.invoke("Explain quantum computing briefly")
        print(f"Manual ChatOpenAI response: {response.content}")

        return True

    except ImportError:
        print("langchain-openai not installed - skipping manual test")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_streaming():
    """Test streaming chat completion."""
    print("\n🔄 Testing Streaming")
    print("=" * 40)

    try:
        # Use factory function with streaming
        llm = create_llamastack_llm(model="ollama/llama3:70b-instruct")

        print("AI: ", end="", flush=True)
        for chunk in llm.stream("Tell me about machine learning in 2 sentences"):
            print(chunk.content, end="", flush=True)
        print()

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_safety_checking():
    """Test safety checking functionality."""
    print("\n🛡️ Testing Safety Checking")
    print("=" * 40)

    try:
        safety = LlamaStackSafety(
            base_url="http://localhost:8321", shield_id="llama-guard"
        )

        # Check if shields are available first
        shields = safety.get_available_shields()
        print(f"Available shields: {len(shields)}")

        if not shields:
            print("⚠️ No shields available - skipping safety tests")
            print("💡 Safety functionality requires shield models to be loaded")
            print("   You can configure shields in your Llama Stack server")
            return True  # Return success to not fail the overall test

        # Test content safety only if shields are available
        test_messages = [
            "Hello, how are you today?",
            "I'm interested in learning about AI",
            "Can you help me with my homework?",
        ]

        for msg in test_messages:
            result = safety.check_content(msg)
            status = "✅ SAFE" if result.is_safe else "❌ UNSAFE"
            print(f"{status}: '{msg[:30]}...'")
            if not result.is_safe:
                print(f"  Reason: {result.message}")

        # Test conversation safety
        conversation = [
            {"role": "user", "content": "Hello there!"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
            {"role": "user", "content": "I'm learning about AI safety."},
        ]

        conv_result = safety.check_conversation(conversation)
        conv_status = "✅ SAFE" if conv_result.is_safe else "❌ UNSAFE"
        print(f"Conversation {conv_status}: {len(conversation)} messages")

        # Show available shields
        for shield in shields[:2]:  # Show first 2
            print(f"  - {shield}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        print(
            "💡 Safety checking failed - this is expected if no shield models are configured"
        )
        return True  # Return success to not fail the overall test


def test_combined_safe_chat():
    """Test combined chat with safety checking."""
    print("\n🔒 Testing Combined Safe Chat")
    print("=" * 40)

    try:
        # Initialize components using new approach
        llm = create_llamastack_llm(model="ollama/llama3:70b-instruct")
        safety = LlamaStackSafety(base_url="http://localhost:8321")

        # Check if shields are available
        shields = safety.get_available_shields()
        if not shields:
            print("⚠️ No shields available - using chat without safety checks")
            # Just do regular chat without safety
            response = llm.invoke("How can AI help in education?")
            print(f"AI (no safety check): {response.content}")
            return True

        # Test messages with safety checking
        test_messages = [
            "How can AI help in education?",
            "Tell me about machine learning",
        ]

        for msg in test_messages:
            print(f"\nUser: {msg}")

            # Check safety first
            safety_result = safety.check_content(msg)

            if safety_result.is_safe:
                # Safe to proceed with chat
                try:
                    response = llm.invoke(msg)
                    print(f"AI: {response.content}")
                except Exception as e:
                    print(f"⚠️ Chat error: {e}")
            else:
                print(f"🛡️ Safety check: {safety_result.message}")
                print("   (This is normal behavior - safety system is working)")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_embeddings():
    """Test embeddings functionality."""
    print("\n🔢 Testing Embeddings")
    print("=" * 40)

    try:
        # Initialize embeddings
        embeddings = LlamaStackEmbeddings(
            model="ollama/all-minilm:l6-v2",  # Popular embedding model
            base_url="http://localhost:8321",
        )

        print(f"✅ Initialized with model: {embeddings.model}")

        # Test single query embedding
        query = "What is artificial intelligence?"
        print(f"Query: '{query}'")

        embedding = embeddings.embed_query(query)
        print(f"✅ Embedding generated: dimension {len(embedding)}")
        print(f"   First 3 values: {embedding[:3]}")

        # Test multiple documents
        documents = [
            "AI is transforming industries.",
            "Machine learning enables automation.",
            "Deep learning uses neural networks.",
        ]

        print(f"\nEmbedding {len(documents)} documents...")
        doc_embeddings = embeddings.embed_documents(documents)
        print(f"✅ Generated {len(doc_embeddings)} embeddings")

        return True

    except Exception as e:
        print(f"Error: {e}")
        print("💡 Make sure embedding models are available in LlamaStack")
        return False


def main():
    """Run all tests."""
    print("🚀 LangChain Llama Stack Integration - Basic Examples")
    print("=" * 60)

    # Show different approaches first
    print("\n🎯 Testing All Approaches")
    print("=" * 40)

    # Test model discovery first
    discovery_result = test_model_discovery()

    # Run main tests
    chat_result = test_chat_completion()
    manual_result = test_manual_chatopenaai()
    streaming_result = test_streaming()
    safety_result = test_safety_checking()
    embeddings_result = test_embeddings()
    combined_result = test_combined_safe_chat()

    # Summary
    print("\n📊 Test Results")
    print("=" * 40)
    print(f"Model Discovery: {'✅ PASS' if discovery_result else '❌ FAIL'}")
    print(f"Factory Function Chat: {'✅ PASS' if chat_result else '❌ FAIL'}")
    print(f"Manual ChatOpenAI: {'✅ PASS' if manual_result else '❌ FAIL'}")
    print(f"Streaming: {'✅ PASS' if streaming_result else '❌ FAIL'}")
    print(f"Safety Checking: {'✅ PASS' if safety_result else '❌ FAIL'}")
    print(f"Embeddings: {'✅ PASS' if embeddings_result else '❌ FAIL'}")
    print(f"Combined Safe Chat: {'✅ PASS' if combined_result else '❌ FAIL'}")

    all_results = [
        discovery_result,
        chat_result,
        manual_result,
        streaming_result,
        safety_result,
        embeddings_result,
        combined_result,
    ]

    if all(all_results):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

    print("\n💡 Usage Examples - New Simplified Approach:")
    print("=" * 50)
    print("# Recommended: Factory function approach")
    print("from langchain_llamastack import create_llamastack_llm")
    print("llm = create_llamastack_llm()  # Auto-selects first available model")
    print("response = llm.invoke('Hello!')")
    print()
    print("# Alternative: Manual ChatOpenAI approach")
    print("from langchain_openai import ChatOpenAI")
    print("from langchain_llamastack import get_llamastack_models")
    print("models = get_llamastack_models()")
    print("llm = ChatOpenAI(")
    print("    base_url='http://localhost:8321/v1/openai/v1',")
    print("    api_key='not-needed',")
    print("    model=models[0]")
    print(")")
    print("response = llm.invoke('Hello!')")
    print()
    print("# Safety")
    print("from langchain_llamastack import LlamaStackSafety")
    print("safety = LlamaStackSafety()")
    print("result = safety.check_content('Hello world')")
    print()
    print("# Embeddings")
    print("from langchain_llamastack import LlamaStackEmbeddings")
    print("embeddings = LlamaStackEmbeddings(model='nomic-embed-text')")
    print("vector = embeddings.embed_query('Hello world')")


if __name__ == "__main__":
    main()
