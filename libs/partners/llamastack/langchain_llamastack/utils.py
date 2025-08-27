"""
Minimal utility functions for LlamaStack operations.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from llama_stack_client import LlamaStackClient
except ImportError:
    LlamaStackClient = None

logger = logging.getLogger(__name__)


def check_llamastack_connection(
    base_url: str = "http://localhost:8321",
    llamastack_api_key: Optional[str] = None,
    model_type: str = "inference",
) -> Dict[str, Any]:
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
            "base_url": base_url,
            "models_count": 0,
            "models": [],
            "error": "llama-stack-client not available. Install with: pip install llama-stack-client",
        }

    try:
        # Create temporary client to check models
        client_kwargs = {"base_url": base_url}
        if llamastack_api_key:
            client_kwargs["api_key"] = llamastack_api_key

        temp_client = LlamaStackClient(**client_kwargs)
        models = temp_client.models.list()

        # Filter models by type
        if model_type == "all":
            model_ids = [model.identifier for model in models]
        else:
            model_ids = []
            for model in models:
                model_id = model.identifier
                model_type_attr = getattr(model, "model_type", None)

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
                    if (
                        model_type_attr == "text_completion"
                        or model_type_attr == "chat_completion"
                        or model_type_attr
                        is None  # Default assumption for untyped models
                        or any(
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
                    ):
                        model_ids.append(model_id)
                elif model_type == "embedding":
                    # Include models that are for embeddings
                    if model_type_attr == "embedding" or any(
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
        def model_priority(model_id):
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
            "base_url": base_url,
            "models_count": len(model_ids),
            "models": model_ids,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Failed to connect to LlamaStack at {base_url}: {e}")
        return {
            "connected": False,
            "base_url": base_url,
            "models_count": 0,
            "models": [],
            "error": str(e),
        }


def list_available_models(
    base_url: str = "http://localhost:8321",
    llamastack_api_key: Optional[str] = None,
    model_type: str = "inference",
) -> List[str]:
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
