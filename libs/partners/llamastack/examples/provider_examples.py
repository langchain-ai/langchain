"""
Example Usage of get_llamastack_providers
=========================================

This example demonstrates how to use the get_llamastack_providers function
to query the actual LlamaStack server for provider configurations.
"""

import json

from langchain_llamastack import (
    ChatLlamaStack,
    check_llamastack_connection,
    get_llamastack_providers,
    list_available_models,
)


def basic_provider_check():
    """Basic example: Check what providers are configured in LlamaStack."""
    print("🔍 Basic Provider Check")
    print("=" * 50)

    # Get providers from default LlamaStack server
    providers = get_llamastack_providers()

    if providers["connected"]:
        print(f"✅ Connected to LlamaStack at {providers['base_url']}")
        print(f"📊 Found {providers['total_inference_providers']} inference providers")
        print(f"🛡️  Found {providers['total_safety_providers']} safety providers")
        print(f"💾 Found {providers['total_memory_providers']} memory providers")

        print("\n📡 Inference Providers:")
        for provider_id, info in providers["inference_providers"].items():
            print(f"  • {provider_id}")
            print(f"    Type: {info['provider_type']}")
            print(f"    API: {info['api_type']}")

        if providers["safety_providers"]:
            print("\n🛡️  Safety Providers:")
            for provider_id, info in providers["safety_providers"].items():
                print(f"  • {provider_id} ({info['provider_type']})")

        if providers["memory_providers"]:
            print("\n💾 Memory Providers:")
            for provider_id, info in providers["memory_providers"].items():
                print(f"  • {provider_id} ({info['provider_type']})")
    else:
        print(f"❌ Cannot connect to LlamaStack: {providers['error']}")


def custom_server_check():
    """Example: Check providers on a custom LlamaStack server."""
    print("\n🌐 Custom Server Check")
    print("=" * 50)

    # Check providers on a custom server
    custom_url = "http://remote-server:8321"  # Example custom server

    providers = get_llamastack_providers(
        base_url=custom_url, llamastack_api_key="your-api-key-if-needed"  # Optional
    )

    if providers["connected"]:
        print(f"✅ Connected to custom LlamaStack at {custom_url}")
        print(f"📊 Total providers: {len(providers['inference_providers'])}")

        # Show detailed provider information
        for provider_id, info in providers["inference_providers"].items():
            print(f"\n📡 Provider: {provider_id}")
            print(f"   Type: {info['provider_type']}")
            print(f"   API Type: {info['api_type']}")
    else:
        print(f"❌ Cannot connect to {custom_url}: {providers['error']}")


def validate_expected_providers():
    """Example: Validate that expected providers are configured."""
    print("\n✅ Provider Validation")
    print("=" * 50)

    # Define expected providers based on your setup
    expected_providers = {
        "ollama": "remote::ollama",
        "together": "remote::together",
        "fireworks": "remote::fireworks",
    }

    providers = get_llamastack_providers()

    if not providers["connected"]:
        print(f"❌ Cannot validate - LlamaStack not available: {providers['error']}")
        return

    print("🔍 Validating expected providers:")

    for expected_id, expected_type in expected_providers.items():
        if expected_id in providers["inference_providers"]:
            actual_type = providers["inference_providers"][expected_id]["provider_type"]
            if actual_type == expected_type:
                print(f"  ✅ {expected_id}: Found with correct type ({actual_type})")
            else:
                print(
                    f"  ⚠️  {expected_id}: Found but wrong type (expected {expected_type}, got {actual_type})"
                )
        else:
            print(f"  ❌ {expected_id}: Not found")

    # Show any unexpected providers
    unexpected = set(providers["inference_providers"].keys()) - set(
        expected_providers.keys()
    )
    if unexpected:
        print(f"\n🆕 Unexpected providers found: {list(unexpected)}")


def troubleshoot_missing_models():
    """Example: Use provider info to troubleshoot missing models."""
    print("\n🔧 Troubleshooting Missing Models")
    print("=" * 50)

    # First, check what models are available
    try:
        models = list_available_models()
        print(f"📋 Available models: {len(models)}")
        if models:
            print(f"   Examples: {models[:3]}...")
        else:
            print("   No models found!")
    except Exception as e:
        print(f"❌ Cannot get models: {e}")
        models = []

    # Then check providers to understand why models might be missing
    providers = get_llamastack_providers()

    if not providers["connected"]:
        print(f"🔍 Root cause: LlamaStack not available - {providers['error']}")
        return

    if not providers["inference_providers"]:
        print("🔍 Root cause: No inference providers configured")
        print("💡 Solution: Configure providers like Ollama, Together AI, etc.")
        return

    print(f"🔍 Found {len(providers['inference_providers'])} inference providers:")
    for provider_id, info in providers["inference_providers"].items():
        print(f"   • {provider_id} ({info['provider_type']})")

    if not models:
        print("💡 Providers are configured but no models available:")
        print("   - Check if provider services are running")
        print("   - Verify API keys are configured")
        print("   - Check provider-specific model setup")


def compare_with_chat_model():
    """Example: Compare provider info with ChatLlamaStack instance."""
    print("\n🔄 Comparison with ChatLlamaStack")
    print("=" * 50)

    # Get provider info directly
    providers = get_llamastack_providers()

    # Create ChatLlamaStack instance
    try:
        llm = ChatLlamaStack(model="meta-llama/Llama-3.1-8B-Instruct")
        chat_info = llm.get_llamastack_info()

        print("📊 Provider Info Comparison:")
        print(
            f"   Direct query: {len(providers.get('inference_providers', {}))} inference providers"
        )
        print(f"   ChatLlamaStack: {chat_info['models_count']} models available")
        print(
            f"   Connection status: {providers.get('connected', False)} vs {chat_info['available']}"
        )

        if providers["connected"] and chat_info["available"]:
            print("✅ Both methods show LlamaStack is working correctly")
        else:
            print("⚠️  Inconsistent results - check configuration")

    except Exception as e:
        print(f"❌ Cannot create ChatLlamaStack: {e}")


def export_provider_config():
    """Example: Export provider configuration for documentation."""
    print("\n📤 Export Provider Configuration")
    print("=" * 50)

    providers = get_llamastack_providers()

    if not providers["connected"]:
        print(f"❌ Cannot export - LlamaStack not available")
        return

    # Create a clean export
    export_data = {
        "llamastack_url": providers["base_url"],
        "connection_status": "connected",
        "providers": {"inference": {}, "safety": {}, "memory": {}},
        "summary": {
            "total_inference_providers": providers["total_inference_providers"],
            "total_safety_providers": providers["total_safety_providers"],
            "total_memory_providers": providers["total_memory_providers"],
        },
    }

    # Add provider details
    for provider_id, info in providers["inference_providers"].items():
        export_data["providers"]["inference"][provider_id] = {
            "type": info["provider_type"],
            "status": "active",
        }

    for provider_id, info in providers["safety_providers"].items():
        export_data["providers"]["safety"][provider_id] = {
            "type": info["provider_type"],
            "status": "active",
        }

    for provider_id, info in providers["memory_providers"].items():
        export_data["providers"]["memory"][provider_id] = {
            "type": info["provider_type"],
            "status": "active",
        }

    print("📋 Provider Configuration:")
    print(json.dumps(export_data, indent=2))


def main():
    """Run all examples."""
    print("🚀 LlamaStack Provider Examples")
    print("=" * 50)

    # Check basic connection first
    status = check_llamastack_connection()
    if not status["connected"]:
        print(f"❌ LlamaStack not available: {status['error']}")
        print("💡 Make sure LlamaStack is running at http://localhost:8321")
        print("   Start with: llama stack run <config-file>")
        return

    # Run all examples
    basic_provider_check()
    # custom_server_check()  # Uncomment if you have a remote server
    validate_expected_providers()
    troubleshoot_missing_models()
    compare_with_chat_model()
    export_provider_config()

    print(f"\n✅ All examples completed!")


if __name__ == "__main__":
    main()
