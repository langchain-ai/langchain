"""
Minimal utility functions for LlamaStack operations.
"""

import logging
from typing import Any, Optional

try:
    from llama_stack_client import LlamaStackClient
except ImportError:
    LlamaStackClient = None  # type: ignore

logger = logging.getLogger(__name__)


def check_llamastack_connection(
    base_url: str = "http://localhost:8321",
    llamastack_api_key: Optional[str] = None,
    model_type: str = "inference",
) -> dict[str, Any]:
    """Check LlamaStack connection and return status information.

    Args:
        base_url: LlamaStack base URL
        llamastack_api_key: LlamaStack API key (if required)
        model_type: Type of models to filter for ('inference', 'embedding', or 'all')

    Returns:
        Dictionary with connection status and model information

    Example:
        status = check_llamastack_connection()
        if status['connected']:
            print(f"Connected! {status['models_count']} models available")
            print(f"Models: {status['models']}")
        else:
            print(f"Connection failed: {status['error']}")
    """
    if LlamaStackClient is None:
        return {
            "connected": False,
            "error": "llama-stack-client not available. I\
            nstall with: pip install llama-stack-client",
            "models": [],
            "models_count": 0,
            "base_url": base_url,
        }

    try:
        # Create temporary client to check models
        if llamastack_api_key:
            temp_client = LlamaStackClient(
                base_url=base_url, api_key=llamastack_api_key
            )
        else:
            temp_client = LlamaStackClient(base_url=base_url)
        models = temp_client.models.list()

        # Filter models by type
        if model_type == "all":
            model_ids = [model.identifier for model in models]
        else:
            model_ids = []
            for model in models:
                try:
                    model_id = getattr(model, "identifier", None)
                    if not model_id:
                        continue
                    model_type_attr = getattr(model, "model_type", None)
                except AttributeError:
                    # Skip models that don't have required attributes
                    continue

                # Filter based on model type or common naming patterns
                if model_type == "inference":
                    # Exclude guard/shield models first (priority exclusion)
                    if any(
                        pattern in model_id.lower()
                        for pattern in [
                            "guard",
                            "shield",
                            "safety",
                            "moderation",
                            "classifier",
                        ]
                    ):
                        continue

                    # Exclude embedding models
                    if any(
                        pattern in model_id.lower()
                        for pattern in [
                            "embed",
                            "embedding",
                            "minilm",
                            "sentence",
                            "bge",
                            "e5",
                        ]
                    ):
                        continue

                    # Include models that are likely for chat/inference
                    model_type_lower = (
                        model_type_attr.lower() if model_type_attr else None
                    )

                    # Check if it matches inference type
                    type_matches = (
                        model_type_lower == "inference"
                        or model_type_lower == "text_completion"
                        or model_type_lower == "chat_completion"
                    )

                    # Check if it matches common inference patterns
                    # (only if no model_type)
                    pattern_matches = (
                        model_type_attr is None  # Only for models without explicit type
                        and any(
                            pattern in model_id.lower()
                            for pattern in [
                                "instruct",  # Most important pattern
                                "chat",
                                "llama",
                                "mistral",
                                "codellama",
                                "vicuna",
                                "alpaca",
                                "qwen",
                                "gemma",
                                "phi",
                                "wizardlm",
                                "orca",
                                "turbo",
                            ]
                        )
                    )

                    if type_matches or pattern_matches:
                        model_ids.append(model_id)
                elif model_type == "embedding":
                    # Include models that are for embeddings
                    model_type_lower = (
                        model_type_attr.lower() if model_type_attr else None
                    )
                    if model_type_lower == "embedding" or any(
                        pattern in model_id.lower()
                        for pattern in [
                            "embed",
                            "embedding",
                            "minilm",
                            "sentence",
                            "bge",
                            "e5",
                        ]
                    ):
                        model_ids.append(model_id)

        # Sort models to prioritize most reliable ones
        def model_priority(model_id: str) -> int:
            """Sort key function to prioritize models."""
            # Fireworks/OpenAI/etc models first (more reliable routing)
            if not model_id.startswith("ollama/"):
                return 0
            # Instruct models within ollama
            if "instruct" in model_id.lower():
                return 1
            # Other ollama models
            return 2

        # Sort models by priority
        model_ids.sort(key=model_priority)

        logger.debug(
            f"Found {len(model_ids)} {model_type} models in LlamaStack: {model_ids}"
        )

        return {
            "connected": True,
            "models": model_ids,
            "models_count": len(model_ids),
            "base_url": base_url,
        }
    except Exception as e:
        logger.error(f"Failed to connect to LlamaStack at {base_url}: {e}")
        return {
            "connected": False,
            "error": str(e),
            "models": [],
            "models_count": 0,
            "base_url": base_url,
        }


def list_available_models(
    base_url: str = "http://localhost:8321",
    llamastack_api_key: Optional[str] = None,
    model_type: str = "inference",
) -> list[str]:
    """List available models from LlamaStack.

    Args:
        base_url: LlamaStack base URL
        llamastack_api_key: LlamaStack API key (if required)
        model_type: Type of models to filter for ('inference', 'embedding', or 'all')

    Returns:
        List of available model names

    Raises:
        ValueError: If LlamaStack is not available
    """
    status = check_llamastack_connection(base_url, llamastack_api_key, model_type)

    if status["connected"]:
        return status["models"]
    else:
        raise ValueError(f"Cannot connect to LlamaStack: {status['error']}")
