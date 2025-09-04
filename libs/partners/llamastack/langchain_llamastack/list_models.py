#!/usr/bin/env python3
"""
Comprehensive script to list LlamaStack available models grouped by type
"""

import os
import sys
from typing import Any, Optional

# Handle imports for both direct execution and module import
try:
    # When run as module
    from .utils import check_llamastack_connection
except ImportError:
    # When run directly - add current directory to path and import
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from utils import check_llamastack_connection  # type: ignore
    except ImportError:
        # If utils is still not found, define a minimal version
        def check_llamastack_connection(
            base_url: str = "http://localhost:8321",
            llamastack_api_key: Optional[str] = None,
            model_type: str = "inference",
        ) -> dict[str, Any]:
            """Fallback function when utils module is not available."""
            return {
                "connected": False,
                "error": "Utils module not available",
                "base_url": base_url,
            }


def classify_model(model_name: str) -> str:
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


def get_detailed_model_info() -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
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

        except Exception:
            # Fallback to simple model list
            return [{"identifier": model} for model in status["models"]], None

    except Exception as e:
        return None, str(e)


def main() -> int:
    try:
        # Get detailed model information
        models, error = get_detailed_model_info()

        if error:
            return 1

        if not models:
            return 1

        # Classify models by type
        model_categories: dict[str, list[dict[str, Any]]] = {
            "inference": [],
            "embedding": [],
            "safety": [],
        }

        for model in models:
            model_name = model["identifier"]
            category = classify_model(model_name)
            model_categories[category].append(model)

        # Display summary
        len(models)

        # Display each category
        for category, display_name, emoji in [
            ("inference", "Inference Models (Chat/Text Generation)", "🤖"),
            ("embedding", "Embedding Models (Vector Representations)", "🔢"),
            ("safety", "Safety Models (Content Moderation)", "🛡️"),
        ]:
            models_in_category = model_categories[category]

            if models_in_category:
                for i, model in enumerate(models_in_category, 1):
                    model_name = model["identifier"]
                    provider_id = model.get("provider_id", "unknown")

                    # Show basic info

                    # Show additional details if available
                    if provider_id != "unknown":
                        pass

                    if (
                        "provider_resource_id" in model
                        and model["provider_resource_id"]
                    ):
                        pass

                    if "model_type" in model and model["model_type"]:
                        pass

            else:
                pass

        # Show provider summary
        providers = set()
        for model in models:
            provider_id = model.get("provider_id", "unknown")
            if provider_id != "unknown":
                providers.add(provider_id)

        if providers:
            for provider in sorted(providers):
                [m for m in models if m.get("provider_id") == provider]

    except Exception:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
