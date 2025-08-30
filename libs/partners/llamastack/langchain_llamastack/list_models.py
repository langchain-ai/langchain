#!/usr/bin/env python3
"""
Comprehensive script to list LlamaStack available models grouped by type
"""

import os
import sys

# Handle imports for both direct execution and module import
try:
    # When run as module
    from .utils import check_llamastack_connection
except ImportError:
    # When run directly - add current directory to path and import
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import check_llamastack_connection


def classify_model(model_name):
    """Classify model based on its name pattern."""
    model_lower = model_name.lower()

    # Safety/Shield models
    if any(
        keyword in model_lower
        for keyword in [
            "guard",
            "shield",
            "safety",
            "moderation",
            "classifier",
            "prompt-guard",
            "llama-guard",
            "code-scanner",
        ]
    ):
        return "safety"

    # Embedding models
    elif any(
        keyword in model_lower
        for keyword in [
            "embed",
            "embedding",
            "vector",
            "retrieval",
            "sentence",
            "all-minilm",
            "e5-",
            "bge-",
            "gte-",
        ]
    ):
        return "embedding"

    # Default to inference (chat/text generation)
    else:
        return "inference"


def get_detailed_model_info():
    """Get detailed model information from LlamaStack."""
    try:
        # Check connection first
        status = check_llamastack_connection()

        if not status["connected"]:
            return None, status["error"]

        # Get models with more details if possible
        try:
            from llama_stack_client import LlamaStackClient

            client = LlamaStackClient(base_url=status["base_url"])
            models = client.models.list()

            detailed_models = []
            for model in models:
                model_info = {
                    "identifier": model.identifier,
                    "provider_id": getattr(model, "provider_id", "unknown"),
                    "provider_resource_id": getattr(
                        model, "provider_resource_id", None
                    ),
                    "model_type": getattr(model, "model_type", None),
                    "metadata": getattr(model, "metadata", {}),
                }
                detailed_models.append(model_info)

            return detailed_models, None

        except Exception as e:
            # Fallback to simple model list
            return [{"identifier": model} for model in status["models"]], None

    except Exception as e:
        return None, str(e)


def main():
    print("🔍 LlamaStack Models by Category")
    print("=" * 50)

    try:
        # Get detailed model information
        models, error = get_detailed_model_info()

        if error:
            print(f"❌ Cannot connect to LlamaStack: {error}")
            print()
            print("💡 Make sure LlamaStack is running:")
            print("  llama stack run ~/.llama/distributions/starter/starter-run.yaml")
            return 1

        if not models:
            print("📭 No models found")
            print()
            print("💡 Possible reasons:")
            print("  • No providers are configured")
            print("  • Providers are configured but not running")
            print("  • Environment variables not set")
            return 1

        # Classify models by type
        model_categories = {"inference": [], "embedding": [], "safety": []}

        for model in models:
            model_name = model["identifier"]
            category = classify_model(model_name)
            model_categories[category].append(model)

        # Display summary
        total_models = len(models)
        print(f"📊 Found {total_models} models total:")
        print(f"   • {len(model_categories['inference'])} inference models")
        print(f"   • {len(model_categories['embedding'])} embedding models")
        print(f"   • {len(model_categories['safety'])} safety models")
        print()

        # Display each category
        for category, display_name, emoji in [
            ("inference", "Inference Models (Chat/Text Generation)", "🤖"),
            ("embedding", "Embedding Models (Vector Representations)", "🔢"),
            ("safety", "Safety Models (Content Moderation)", "🛡️"),
        ]:
            models_in_category = model_categories[category]

            if models_in_category:
                print(f"{emoji} {display_name}")
                print("-" * len(f"{emoji} {display_name}"))

                for i, model in enumerate(models_in_category, 1):
                    model_name = model["identifier"]
                    provider_id = model.get("provider_id", "unknown")

                    # Show basic info
                    print(f"  {i}. {model_name}")

                    # Show additional details if available
                    if provider_id != "unknown":
                        print(f"     Provider: {provider_id}")

                    if (
                        "provider_resource_id" in model
                        and model["provider_resource_id"]
                    ):
                        print(f"     Resource ID: {model['provider_resource_id']}")

                    if "model_type" in model and model["model_type"]:
                        print(f"     Type: {model['model_type']}")

                print()
            else:
                print(f"{emoji} {display_name}")
                print("-" * len(f"{emoji} {display_name}"))
                print("  No models found in this category")
                print()

        # Show provider summary
        providers = set()
        for model in models:
            provider_id = model.get("provider_id", "unknown")
            if provider_id != "unknown":
                providers.add(provider_id)

        if providers:
            print("📡 Active Providers:")
            for provider in sorted(providers):
                provider_models = [
                    m for m in models if m.get("provider_id") == provider
                ]
                print(f"  • {provider}: {len(provider_models)} models")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
