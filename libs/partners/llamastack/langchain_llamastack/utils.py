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
) -> Dict[str, Any]:
    """Check LlamaStack connection and return status information.

    Args:
        base_url: LlamaStack base URL
        llamastack_api_key: LlamaStack API key (if required)

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
        model_ids = [model.identifier for model in models]

        logger.debug(f"Found {len(model_ids)} models in LlamaStack: {model_ids}")

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
) -> List[str]:
    """List available models from LlamaStack.

    Args:
        base_url: LlamaStack base URL
        llamastack_api_key: LlamaStack API key (if required)

    Returns:
        List of available model names

    Raises:
        ValueError: If LlamaStack is not available
    """
    status = check_llamastack_connection(base_url, llamastack_api_key)

    if status["connected"]:
        return status["models"]
    else:
        raise ValueError(f"Cannot connect to LlamaStack: {status['error']}")
