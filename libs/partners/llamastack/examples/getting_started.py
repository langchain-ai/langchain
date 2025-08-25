#!/usr/bin/env python3
"""
Complete Getting Started Guide for LangChain LlamaStack
=======================================================

This guide shows you how to use the LangChain LlamaStack package immediately after installation.
Run each example independently to test different features.
"""

# =============================================================================
# INSTALLATION & PROVIDER SETUP GUIDE
# =============================================================================

installation_guide = """
# Installation & Provider Setup Guide
====================================

1. Install the package:
   pip install -e /path/to/langchain_llamastack_integration/langchain/libs/partners/llamastack

2. Set up provider environment variables (choose your providers):

   # For Ollama (local models)
   export OLLAMA_BASE_URL="http://localhost:11434"

   # For OpenAI
   export OPENAI_API_KEY="your-openai-api-key"

   # For Together AI
   export TOGETHER_API_KEY="your-together-api-key"

   # For Fireworks AI
   export FIREWORKS_API_KEY="your-fireworks-api-key"

   # For Anthropic
   export ANTHROPIC_API_KEY="your-anthropic-api-key"

   # For Groq
   export GROQ_API_KEY="your-groq-api-key"

3. Start your providers:
   # For Ollama
   ollama serve

   # For vLLM (if using local vLLM)
   python -m vllm.entrypoints.openai.api_server --model your-model

4. Pull models (for Ollama):
   ollama pull llama3:8b
   ollama pull ollama/nomic-embed-text:l6-v2
   ollama pull shieldgemma:2b

5. Start LlamaStack server with your providers:
   # With Ollama only
   llama-stack-run --port 8321 --inference-provider remote::ollama

   # With multiple providers
   llama-stack-run --port 8321 --inference-provider remote::ollama --inference-provider remote::together

6. Verify setup:
   curl http://localhost:8321/v1/models

7. Check environment variables:
   # List all API keys
   env | grep API_KEY

   # Check specific providers
   echo $OPENAI_API_KEY
   echo $TOGETHER_API_KEY
"""

print(installation_guide)

# =============================================================================
# EXAMPLE 1: BASIC CHAT COMPLETION
# =============================================================================


def example_1_basic_chat():
    """Example 1: Basic chat completion."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Chat Completion")
    print("=" * 60)

    try:
        from langchain_llamastack import ChatLlamaStack

        # Initialize the chat model
        llm = ChatLlamaStack(
            model="ollama/llama3:8b",  # Use your available model
            base_url="http://localhost:8321",
        )

        # Basic chat
        print("🤖 Asking: 'What is artificial intelligence?'")
        response = llm.invoke("What is artificial intelligence?")
        print(f"🤖 Response: {response.content}")

        # Multi-turn conversation
        print("\n💬 Multi-turn conversation:")
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello! What's your name?"},
        ]

        response = llm.invoke(messages)
        print(f"🤖 AI: {response.content}")

        print("✅ Example 1 completed successfully!")

    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
        print("💡 Make sure LlamaStack server is running and the model is available")


# =============================================================================
# EXAMPLE 2: STREAMING CHAT
# =============================================================================


def example_2_streaming_chat():
    """Example 2: Streaming chat responses."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Streaming Chat")
    print("=" * 60)

    try:
        from langchain_llamastack import ChatLlamaStack

        # Initialize with streaming enabled
        llm = ChatLlamaStack(
            model="ollama/llama3:8b",
            base_url="http://localhost:8321",
            streaming=True,
        )

        print("🔄 Streaming response for: 'Tell me a short story about AI'")
        print("🤖 AI: ", end="", flush=True)

        # Stream the response
        for chunk in llm.stream("Tell me a short story about AI"):
            print(chunk.content, end="", flush=True)

        print("\n✅ Example 2 completed successfully!")

    except Exception as e:
        print(f"❌ Example 2 failed: {e}")


# =============================================================================
# EXAMPLE 3: EMBEDDINGS
# =============================================================================


def example_3_embeddings():
    """Example 3: Text embeddings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Text Embeddings")
    print("=" * 60)

    try:
        from langchain_llamastack import LlamaStackEmbeddings

        # Initialize embeddings
        embeddings = LlamaStackEmbeddings(
            model="ollama/nomic-embed-text",  # Popular embedding model
            base_url="http://localhost:8321",
        )

        # Single text embedding
        text = "Hello, world!"
        print(f"📝 Embedding text: '{text}'")

        embedding = embeddings.embed_query(text)
        print(f"✅ Generated embedding with dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")

        # Multiple documents
        documents = [
            "Artificial intelligence is transforming industries.",
            "Machine learning enables computers to learn from data.",
            "Natural language processing helps understand text.",
        ]

        print(f"\n📚 Embedding {len(documents)} documents...")
        doc_embeddings = embeddings.embed_documents(documents)
        print(f"✅ Generated {len(doc_embeddings)} embeddings")

        # Check available models
        available_models = embeddings.get_available_models()
        print(f"📋 Available embedding models: {len(available_models)}")
        for model in available_models[:3]:  # Show first 3
            print(f"   - {model}")

        print("✅ Example 3 completed successfully!")

    except Exception as e:
        print(f"❌ Example 3 failed: {e}")


# =============================================================================
# EXAMPLE 4: SAFETY CHECKING
# =============================================================================


def example_4_safety_checking():
    """Example 4: Content safety checking."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Safety Checking")
    print("=" * 60)

    try:
        from langchain_llamastack import LlamaStackSafety

        # Initialize safety checker (uses Ollama shield models)
        safety = LlamaStackSafety(
            base_url="http://localhost:8321", shield_id="code-scanner"
        )

        # Check available shields
        shields = safety.list_available_shields()
        print(f"🛡️ Available shield models: {len(shields)}")

        if not shields:
            print("⚠️ No shield models found. Install with:")
            print("   ollama pull shieldgemma:2b")
            return

        for shield in shields:
            print(f"   - {shield}")

        # Test content safety
        test_contents = [
            "Hello, how are you today?",
            "Tell me about machine learning",
            "Can you help me with homework?",
        ]

        print(f"\n🧪 Testing {len(test_contents)} pieces of content:")

        for content in test_contents:
            print(f"\n📝 Testing: '{content[:40]}...'")
            result = safety.check_content(content)

            status = "✅ SAFE" if result.is_safe else "❌ UNSAFE"
            print(f"   {status}: {result.message}")
            print(f"   Shield: {result.shield_id}")

        print("✅ Example 4 completed successfully!")

    except Exception as e:
        print(f"❌ Example 4 failed: {e}")
        print("💡 Make sure Ollama is running and shield models are installed")


# =============================================================================
# EXAMPLE 5: COMBINED WORKFLOW
# =============================================================================


def example_5_combined_workflow():
    """Example 5: Complete workflow with chat, embeddings, and safety."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Complete Workflow")
    print("=" * 60)

    try:
        from langchain_llamastack import (
            ChatLlamaStack,
            LlamaStackEmbeddings,
            LlamaStackSafety,
        )

        # Initialize all components
        llm = ChatLlamaStack(model="ollama/llama3:8b", base_url="http://localhost:8321")

        embeddings = LlamaStackEmbeddings(
            model="ollama/nomic-embed-text", base_url="http://localhost:8321"
        )

        safety = LlamaStackSafety(
            base_url="http://localhost:8321", shield_id="code-scanner"
        )

        print("✅ All components initialized successfully!")

        # Workflow: Safe chat with context retrieval
        user_query = "What are the benefits of renewable energy?"
        print(f"\n👤 User query: '{user_query}'")

        # Step 1: Safety check
        print("\n🛡️ Step 1: Safety check...")
        safety_result = safety.check_content(user_query)

        if not safety_result.is_safe:
            print(f"❌ Query rejected: {safety_result.message}")
            return

        print("✅ Query passed safety check")

        # Step 2: Generate embedding for semantic search (if you had a knowledge base)
        print("\n🔍 Step 2: Generate query embedding...")
        query_embedding = embeddings.embed_query(user_query)
        print(f"✅ Generated embedding (dimension: {len(query_embedding)})")

        # Step 3: Get AI response
        print("\n🤖 Step 3: Generate AI response...")
        response = llm.invoke(user_query)
        print(f"✅ AI Response: {response.content[:200]}...")

        # Step 4: Check response safety
        print("\n🛡️ Step 4: Check response safety...")
        response_safety = safety.check_content(response.content)

        if response_safety.is_safe:
            print("✅ Response passed safety check")
        else:
            print(f"⚠️ Response flagged: {response_safety.message}")

        print("✅ Example 5 completed successfully!")

    except Exception as e:
        print(f"❌ Example 5 failed: {e}")


# =============================================================================
# EXAMPLE 6: RAG-STYLE SEMANTIC SEARCH
# =============================================================================


def example_6_semantic_search():
    """Example 6: Semantic search with embeddings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Semantic Search")
    print("=" * 60)

    try:
        from langchain_llamastack import LlamaStackEmbeddings

        embeddings = LlamaStackEmbeddings(
            model="ollama/nomic-embed-text", base_url="http://localhost:8321"
        )

        # Sample knowledge base
        knowledge_base = [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning algorithms can automatically learn patterns from data.",
            "Neural networks are inspired by the structure of the human brain.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret and analyze visual information.",
            "Deep learning is a subset of machine learning using multi-layer neural networks.",
        ]

        print(f"📚 Knowledge base with {len(knowledge_base)} documents")

        # User query
        query = "What is artificial intelligence?"
        print(f"🔍 User query: '{query}'")

        # Perform semantic search
        print("\n🧠 Performing semantic search...")

        try:
            similar_docs = embeddings.similarity_search_by_vector(
                embeddings.embed_query(query), knowledge_base, k=3
            )

            print("📊 Most relevant documents:")
            for i, (doc, score) in enumerate(similar_docs, 1):
                print(f"  {i}. Score: {score:.3f}")
                print(f"     Text: {doc}")

        except ImportError:
            print("⚠️ Install numpy and scikit-learn for similarity search:")
            print("   pip install numpy scikit-learn")

            # Fallback: just show embeddings
            print("📊 Generated embeddings for query and documents")
            query_emb = embeddings.embed_query(query)
            doc_embs = embeddings.embed_documents(knowledge_base[:2])
            print(f"   Query embedding dimension: {len(query_emb)}")
            print(f"   Document embeddings: {len(doc_embs)} x {len(doc_embs[0])}")

        print("✅ Example 6 completed successfully!")

    except Exception as e:
        print(f"❌ Example 6 failed: {e}")


# =============================================================================
# EXAMPLE 7: LANGCHAIN INTEGRATION
# =============================================================================


def example_7_langchain_integration():
    """Example 7: Using with LangChain components."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: LangChain Integration")
    print("=" * 60)

    try:
        from langchain.schema import HumanMessage, SystemMessage
        from langchain_llamastack import ChatLlamaStack, LlamaStackEmbeddings

        # Initialize components
        llm = ChatLlamaStack(model="ollama/llama3:8b", base_url="http://localhost:8321")

        embeddings = LlamaStackEmbeddings(
            model="ollama/nomic-embed-text", base_url="http://localhost:8321"
        )

        # Example 1: Using with LangChain message types
        print("💬 Using LangChain message types...")

        messages = [
            SystemMessage(
                content="You are a helpful AI assistant specializing in technology."
            ),
            HumanMessage(content="Explain quantum computing in simple terms."),
        ]

        response = llm.invoke(messages)
        print(f"✅ Response: {response.content[:150]}...")

        # Example 2: Using embeddings with custom similarity
        print("\n🔍 Custom similarity calculation...")

        texts = ["quantum computing", "classical computing", "artificial intelligence"]
        text_embeddings = embeddings.embed_documents(texts)

        print(f"✅ Generated embeddings for {len(texts)} texts")
        print(f"   Embedding dimensions: {len(text_embeddings[0])}")

        # Example 3: Batch processing
        print("\n📦 Batch processing multiple queries...")

        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does AI work?",
        ]

        print("Processing queries:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
            response = llm.invoke(query)
            print(f"     Response: {response.content[:50]}...")

        print("✅ Example 7 completed successfully!")

    except Exception as e:
        print(f"❌ Example 7 failed: {e}")


# =============================================================================
# QUICK START TEMPLATES
# =============================================================================


def show_quick_start_templates():
    """Show quick start code templates."""
    print("\n" + "=" * 60)
    print("QUICK START TEMPLATES")
    print("=" * 60)

    templates = {
        "Basic Chat": """
from langchain_llamastack import ChatLlamaStack

llm = ChatLlamaStack(
    model="ollama/llama3:8b",
    base_url="http://localhost:8321"
)

response = llm.invoke("Hello, world!")
print(response.content)
""",
        "Streaming Chat": """
from langchain_llamastack import ChatLlamaStack

llm = ChatLlamaStack(
    model="ollama/llama3:8b",
    base_url="http://localhost:8321",
    streaming=True
)

for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
""",
        "Embeddings": """
from langchain_llamastack import LlamaStackEmbeddings

embeddings = LlamaStackEmbeddings(
    model="ollama/nomic-embed-text",
    base_url="http://localhost:8321"
)

# Single embedding
embedding = embeddings.embed_query("Hello world")
print(f"Dimension: {len(embedding)}")

# Multiple documents
docs = ["Doc 1", "Doc 2", "Doc 3"]
doc_embeddings = embeddings.embed_documents(docs)
print(f"Generated {len(doc_embeddings)} embeddings")
""",
        "Safety Checking": """
from langchain_llamastack import LlamaStackSafety

safety = LlamaStackSafety(
    base_url="http://localhost:8321",
    shield_id="code-scanner"
)

result = safety.check_content("Hello world")
print(f"Safe: {result.is_safe}")
print(f"Message: {result.message}")
""",
        "Complete Workflow": """
from langchain_llamastack import ChatLlamaStack, LlamaStackEmbeddings, LlamaStackSafety

# Initialize components
llm = ChatLlamaStack(model="ollama/llama3:8b", base_url="http://localhost:8321")
embeddings = LlamaStackEmbeddings(model="ollama/nomic-embed-text", base_url="http://localhost:8321")
safety = LlamaStackSafety(base_url="http://localhost:8321", shield_id="code-scanner")

# Safe AI workflow
user_input = "What is AI?"

# 1. Check safety
if safety.check_content(user_input).is_safe:
    # 2. Generate response
    response = llm.invoke(user_input)

    # 3. Check response safety
    if safety.check_content(response.content).is_safe:
        print(response.content)
    else:
        print("Response blocked by safety filter")
else:
    print("Input blocked by safety filter")
""",
    }

    for name, code in templates.items():
        print(f"\n📋 {name}:")
        print("-" * 40)
        print(code)


# =============================================================================
# TROUBLESHOOTING GUIDE
# =============================================================================


def show_troubleshooting():
    """Show troubleshooting guide."""
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING")
    print("=" * 60)

    troubleshooting = """
Common Issues and Solutions:

1. **Connection Error**: "Failed to connect to LlamaStack"
   - Check LlamaStack server: curl http://localhost:8321/v1/models
   - Make sure that the llama stack server is up and running at port 8321 by checking curl http://localhost:8321/
   - Check port and URL

2. **Model Not Found**: "Model 'xyz' not found"
   - List available models: curl http://localhost:8321/v1/models
   - Pull model in Ollama: ollama pull llama3:8b
   - Use correct model identifier

3. **Import Error**: "No module named 'langchain_llamastack'"
   - Install package: pip install -e /path/to/package
   - Check Python path
   - Verify installation: python -c "import langchain_llamastack"

4. **Safety Issues**: "No shield models available"
   - Install shield models: ollama pull shieldgemma:2b
   - Check Ollama: ollama list
   - Verify Ollama is running: ollama serve
   - export SAFETY_MODEL="llama-guard"
   - export CODE_SCANNER_MODEL="code-scanner"

5. **Embeddings Error**: "No embedding models"
   - Install embedding models: ollama pull ollama/nomic-embed-text:l6-v2 (if you are running ollama locally)
   - Run VLLM server locally or connect remotely to together ai or to a different provider.
   - Check model type in LlamaStack
   - Verify model identifier

6. **Timeout Errors**: "Request timeout"
   - Increase timeout: request_timeout=60.0
   - Check server load
   - Try smaller requests

Quick Verification Commands:
- curl http://localhost:8321/v1/models (LlamaStack models)
- curl http://localhost:11434/api/tags (Ollama models)
- ollama list (Local Ollama models)
- ps aux | grep llama (Check running processes)
"""

    print(troubleshooting)


def list_and_discover_models():
    """Show how to list models from different sources."""
    print("\n" + "=" * 60)
    print("MODEL LISTING & DISCOVERY")
    print("=" * 60)

    try:
        from langchain_llamastack import ChatLlamaStack

        print("📋 Discovering models from all sources...")

        # List models from LlamaStack
        print("\n🔍 Models from LlamaStack server:")
        try:
            llm = ChatLlamaStack(base_url="http://localhost:8321")
            llamastack_models = llm.get_available_models()

            if llamastack_models:
                print(f"✅ Found {len(llamastack_models)} models in LlamaStack:")

                # Group models by provider
                providers = {}
                for model in llamastack_models:
                    if "/" in model:
                        provider = model.split("/")[0]
                        if provider not in providers:
                            providers[provider] = []
                        providers[provider].append(model)
                    else:
                        if "other" not in providers:
                            providers["other"] = []
                        providers["other"].append(model)

                for provider, models in providers.items():
                    print(f"\n   📁 {provider.upper()}:")
                    for model in models[:5]:  # Show first 5 per provider
                        print(f"      - {model}")
                    if len(models) > 5:
                        print(f"      ... and {len(models) - 5} more")

            else:
                print("❌ No models found in LlamaStack")
                print(
                    "💡 Make sure LlamaStack server is running and providers are configured"
                )

        except Exception as e:
            print(f"❌ Failed to connect to LlamaStack: {e}")
            print(
                "💡 Start LlamaStack server: llama-stack-run --port 8321 --inference-provider remote::ollama"
            )

        # Show how to use different models
        print("\n💡 How to use different models:")
        print("-" * 40)

        examples = [
            ("Ollama Local", "ChatLlamaStack(model='ollama/llama3:8b')"),
            (
                "Together AI",
                "ChatLlamaStack(model='together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo')",
            ),
            ("OpenAI", "ChatLlamaStack(model='openai/gpt-4o-mini')"),
            (
                "Fireworks",
                "ChatLlamaStack(model='fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct')",
            ),
        ]

        for name, code in examples:
            print(f"   {name}: {code}")

        print("\n🔍 Manual verification commands:")
        print("-" * 40)
        print("   # Check LlamaStack models")
        print("   curl http://localhost:8321/v1/models")
        print()
        print("   # Check Ollama models")
        print("   curl http://localhost:11434/api/tags")
        print()
        print("   # List Ollama models")
        print("   ollama list")

        return True

    except Exception as e:
        print(f"❌ Model listing failed: {e}")
        return False


def validate_provider_connections():
    """Check and validate provider configurations."""
    print("\n" + "=" * 60)
    print("PROVIDER CONNECTION VALIDATION")
    print("=" * 60)

    try:
        print("🔍 Validating provider connections...")

        # Test connections to different providers
        providers_to_test = [
            ("Ollama", "http://localhost:11434/api/tags", "Local Ollama server"),
            ("LlamaStack", "http://localhost:8321/v1/models", "LlamaStack server"),
        ]

        import httpx

        for name, url, description in providers_to_test:
            print(f"\n📡 Testing {name} ({description}):")
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(url)
                    if response.status_code == 200:
                        print(f"   ✅ {name} is accessible")

                        # Parse response for model count
                        try:
                            data = response.json()
                            if name == "Ollama" and "models" in data:
                                model_count = len(data["models"])
                                print(f"   📋 {model_count} models available in Ollama")
                                if model_count == 0:
                                    print(
                                        f"      💡 Pull models: ollama pull llama3:8b"
                                    )
                            elif name == "LlamaStack" and "data" in data:
                                model_count = len(data["data"])
                                print(
                                    f"   📋 {model_count} models available in LlamaStack"
                                )
                                if model_count == 0:
                                    print(f"      💡 Configure providers in LlamaStack")
                        except:
                            print(f"   📋 Connected but couldn't parse model data")
                    else:
                        print(f"   ❌ {name} returned status {response.status_code}")

            except httpx.ConnectError:
                print(f"   ❌ {name} is not running or not accessible")
                if name == "Ollama":
                    print(f"      💡 Start with: ollama serve")
                elif name == "LlamaStack":
                    print(
                        f"      💡 Start with: llama-stack-run --port 8321 --inference-provider remote::ollama"
                    )
            except Exception as e:
                print(f"   ❌ {name} check failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Provider validation failed: {e}")
        return False


# =============================================================================
# MAIN FUNCTION
# =============================================================================


# Update the main function to call the complete implementation
def main():
    """Run all examples."""
    print("🚀 LangChain LlamaStack - Complete Usage Examples")
    print("=" * 70)
    print(
        "This guide shows you how to use LangChain LlamaStack immediately after installation."
    )
    print("Each example can be run independently.")

    examples = [
        ("Provider Environment Check", check_provider_environment),
        ("Model Listing & Discovery", list_and_discover_models),
        ("Provider Connection Validation", validate_provider_connections),
        ("Basic Chat Completion", example_1_basic_chat),
        ("Streaming Chat", example_2_streaming_chat),
        ("Text Embeddings", example_3_embeddings),
        ("Safety Checking", example_4_safety_checking),
        ("Combined Workflow", example_5_combined_workflow),
        ("Semantic Search", example_6_semantic_search),
        ("LangChain Integration", example_7_langchain_integration),
    ]

    print(f"\n🎯 Running {len(examples)} examples:")

    results = {}
    for name, example_func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            example_func()
            results[name] = True
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = False

    # Show templates and troubleshooting
    # show_quick_start_templates()
    show_troubleshooting()

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<25} {status}")

    print(f"\nOverall: {passed}/{total} examples passed")

    if passed == total:
        print("🎉 All examples working! You're ready to use LangChain LlamaStack!")
    elif passed > 0:
        print("⚠️ Some examples working. Check failed examples above.")
    else:
        print("❌ No examples working. Check your setup:")
        print("   1. LlamaStack server running?")
        print("   2. Ollama running with models?")
        print("   3. Package installed correctly?")

    print("\n🎯 Next Steps:")
    print("- Copy any template code above to get started")
    print("- Read the troubleshooting guide if you have issues")
    print("- Explore advanced features in the documentation")
    print("- Build your own AI applications!")


if __name__ == "__main__":
    main()
