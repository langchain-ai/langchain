#!/usr/bin/env python3
"""
Test the LlamaStack Embeddings implementation.
"""

from langchain_llamastack import LlamaStackEmbeddings


def test_basic_embeddings():
    """Test basic embeddings functionality."""
    print("🔢 Testing Basic Embeddings")
    print("=" * 50)

    try:
        # Initialize embeddings
        embeddings = LlamaStackEmbeddings(
            model="ollama/nomic-embed-text",  # Popular embedding model
            base_url="http://localhost:8321",
        )

        print(f"✅ Initialized with model: {embeddings.model}")

        # Test single query embedding
        print("\n📝 Testing Single Query Embedding")
        print("-" * 30)

        query = "What is artificial intelligence?"
        print(f"Query: '{query}'")

        embedding = embeddings.embed_query(query)
        print(f"✅ Embedding generated: dimension {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_document_embeddings():
    """Test document embeddings functionality."""
    print("\n📚 Testing Document Embeddings")
    print("=" * 50)

    try:
        embeddings = LlamaStackEmbeddings(
            model="ollama/nomic-embed-text", base_url="http://localhost:8321"
        )

        # Test multiple documents
        documents = [
            "Artificial intelligence is transforming many industries.",
            "Machine learning enables computers to learn from data.",
            "Deep learning uses neural networks with multiple layers.",
        ]

        print(f"Embedding {len(documents)} documents...")

        doc_embeddings = embeddings.embed_documents(documents)

        print(f"✅ Generated {len(doc_embeddings)} embeddings")
        print(f"   Each embedding has dimension: {len(doc_embeddings[0])}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_available_models():
    """Test getting available embedding models."""
    print("\n📋 Testing Available Models")
    print("=" * 50)

    try:
        embeddings = LlamaStackEmbeddings(base_url="http://localhost:8321")

        available_models = embeddings.get_available_models()
        print(f"✅ Found {len(available_models)} embedding models:")

        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}. {model}")

        return len(available_models) > 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run embedding tests."""
    print("🚀 LangChain LlamaStack Embeddings - Test Suite")
    print("=" * 70)

    tests = [
        ("Basic Embeddings", test_basic_embeddings),
        ("Document Embeddings", test_document_embeddings),
        ("Available Models", test_available_models),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n📊 Test Summary")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Embeddings integration is working.")
    else:
        print("⚠️ Some tests failed. Check LlamaStack server and model availability.")


if __name__ == "__main__":
    main()
