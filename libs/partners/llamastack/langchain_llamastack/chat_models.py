"""LangChain chat model integration for Llama Stack using ChatOpenAI."""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:
    ChatOpenAI = None  # type: ignore


logger = logging.getLogger(__name__)


def check_llamastack_connection(
    base_url: str, model_type: str = "llm"
) -> dict[str, Any]:
    """Check LlamaStack connection and return available models."""
    try:
        from llama_stack_client import LlamaStackClient

        client = LlamaStackClient(base_url=base_url)
        models = client.models.list()

        # Filter models by type (LlamaStack uses "llm" not "inference")
        model_ids = []
        for model in models:
            if hasattr(model, "identifier") and hasattr(model, "model_type"):
                if model.model_type == model_type:
                    model_ids.append(model.identifier)

        return {
            "connected": True,
            "models": model_ids,
            "models_count": len(model_ids),
            "base_url": base_url,
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "models": [],
            "models_count": 0,
            "base_url": base_url,
        }


def list_available_models(base_url: str, model_type: str = "llm") -> list[str]:
    """List available models from LlamaStack."""
    try:
        result = check_llamastack_connection(base_url, model_type)
        return result.get("models", [])
    except Exception:
        return []


def create_llamastack_llm(
    model: Optional[str] = None,
    base_url: str = "http://localhost:8321",
    auto_fallback: Optional[bool] = None,  # Auto-determine based on whether
    # model is specified
    validate_model: bool = True,
    provider_api_keys: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for LlamaStack.
    Args:
        model: Model name to use. If None, auto-selects first available model.
        base_url: LlamaStack base URL (without /v1/openai/v1 suffix).
        auto_fallback: Whether to automatically fallback to available models.
        validate_model: Whether to test model accessibility before using.
        provider_api_keys: Dict of provider API \
        keys (e.g., {"fireworks": "api-key", "together": "api-key"}).
        **kwargs: Additional arguments passed to ChatOpenAI.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ConnectionError: If LlamaStack is not available.
        ValueError: If no models are available.
        ImportError: If langchain-openai is not installed.

    Examples:
        # Simple usage - auto-select first available model
        llm = create_llamastack_llm()

        # Specify a model with auto-fallback
        llm = create_llamastack_llm(model="llama3.1:8b")

        # With provider API keys for Fireworks
        llm = create_llamastack_llm(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            provider_api_keys={"fireworks": "your-fireworks-api-key"}
        )

        # With multiple provider keys
        llm = create_llamastack_llm(
            provider_api_keys={
                "fireworks": "fw-key",
                "together": "together-key",
                "openai": "openai-key"
            }
        )

        # Pass additional ChatOpenAI parameters
        llm = create_llamastack_llm(model="ollama/llama3:70b-instruct")
    """
    if ChatOpenAI is None:
        raise ImportError(
            "langchain-openai is required for LlamaStack integration. "
            "Install with: pip install langchain-openai"
        )

    # Check LlamaStack connection and get available models
    status = check_llamastack_connection(base_url)
    if not status["connected"]:
        raise ConnectionError(
            f"LlamaStack not available at {base_url}: {status['error']}\n"
            f"Make sure LlamaStack is running and accessible."
        )

    available_models = status["models"]
    if not available_models:
        raise ValueError(
            f"No models available in LlamaStack at {base_url}. "
            f"Make sure LlamaStack is configured with providers and models."
        )

    # Auto-determine fallback behavior if not explicitly set
    if auto_fallback is None:
        if model is None:
            auto_fallback = True  # Auto-selection mode: enable fallback for convenience
        else:
            auto_fallback = False  # Explicit model: strict behavior by default

    logger.debug(
        f"Using auto_fallback={auto_fallback} \
        (model={'None' if model is None else f'explicit: {model}'})"
    )

    # Handle model selection with validation
    selected_model = model
    if selected_model is None:
        selected_model = _find_working_model(available_models, base_url, validate_model)
        logger.info(f"Auto-selected model: {selected_model}")
    elif auto_fallback and selected_model not in available_models:
        logger.warning(
            f"Model '{selected_model}' not available. "
            f"Available models: {available_models}"
        )
        selected_model = _find_working_model(available_models, base_url, validate_model)
        logger.info(f"Falling back to: {selected_model}")
    elif not auto_fallback and selected_model not in available_models:
        raise ValueError(
            f"Model '{selected_model}' not found in LlamaStack. "
            f"Available models: {available_models}"
        )
    elif validate_model:
        # Validate the explicitly requested model
        if not _test_model_accessibility(selected_model, base_url):
            if auto_fallback:
                logger.warning(
                    f"Model '{selected_model}' not accessible, finding alternative..."
                )
                selected_model = _find_working_model(
                    available_models, base_url, validate_model
                )
                logger.info(f"Falling back to: {selected_model}")
            else:
                raise ValueError(
                    f"Model '{selected_model}' is listed but not accessible. "
                    f"This usually means the provider is not\
                     configured in LlamaStack routing table."
                )

    # Configure ChatOpenAI for LlamaStack
    openai_kwargs = {
        "base_url": f"{base_url}/v1/openai/v1",
        "api_key": kwargs.pop(
            "api_key", "not-needed"
        ),  # LlamaStack doesn't require real API keys
        "model": selected_model,
        **kwargs,
    }

    # Handle provider API keys by setting up default headers
    default_headers = _get_provider_headers(selected_model, provider_api_keys)
    if default_headers:
        openai_kwargs["default_headers"] = default_headers
        logger.info(f"Added provider headers for model: {selected_model}")

    logger.info(f"Creating ChatOpenAI instance for LlamaStack model: {selected_model}")
    return ChatOpenAI(**openai_kwargs)


def _get_provider_headers(
    model: str, provider_api_keys: Optional[dict[str, str]] = None
) -> dict[str, str]:
    """Get the appropriate headers for provider API keys based on the model."""
    import json
    import os

    headers = {}

    # Initialize provider_api_keys if None
    if provider_api_keys is None:
        provider_api_keys = {}

    # Detect provider from model name and get API key
    provider_data = {}

    logger.debug(f"Checking provider for model: {model}")

    # Check all possible providers
    # (don't use elif - a model might match multiple patterns)

    # Fireworks models
    if (
        "fireworks" in model.lower()
        or "accounts/fireworks" in model
        or "fw-" in model.lower()
    ):
        api_key = provider_api_keys.get("fireworks") or os.getenv("FIREWORKS_API_KEY")
        if api_key:
            provider_data["fireworks_api_key"] = api_key
            logger.info(f"Added Fireworks API key for model: {model}")
        else:
            logger.warning(
                f"Fireworks model detected ({model}) \
                but no API key found. Set FIREWORKS_API_KEY environment variable."
            )

    # Together models
    if "together" in model.lower() or "togethercomputer" in model.lower():
        api_key = provider_api_keys.get("together") or os.getenv("TOGETHER_API_KEY")
        if api_key:
            provider_data["together_api_key"] = api_key
            logger.info(f"Added Together API key for model: {model}")

    # OpenAI models
    if "gpt-" in model.lower() or "openai" in model.lower() or model.startswith("gpt"):
        api_key = provider_api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        if api_key:
            provider_data["openai_api_key"] = api_key
            logger.info(f"Added OpenAI API key for model: {model}")

    # Anthropic models
    if "claude" in model.lower() or "anthropic" in model.lower():
        api_key = provider_api_keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            provider_data["anthropic_api_key"] = api_key
            logger.info(f"Added Anthropic API key for model: {model}")

    # Add provider data header if we have any provider keys
    if provider_data:
        headers["X-LlamaStack-Provider-Data"] = json.dumps(provider_data)
        logger.info(f"Set provider data header: {json.dumps(provider_data, indent=2)}")
    else:
        logger.debug(f"No provider API keys needed for model: {model}")
    return headers


def _find_working_model(
    available_models: list[str], base_url: str, validate: bool = True
) -> str:
    """Find the first working model from the available models list."""
    if not validate:
        return available_models[0]

    for model in available_models:
        if _test_model_accessibility(model, base_url):
            return model

    # If no models work, return the first one (let it fail naturally with helpful error)
    logger.warning("No models passed accessibility test, using first available model")
    return available_models[0]


def _test_model_accessibility(model: str, base_url: str) -> bool:
    """Test if a model is actually accessible via the OpenAI-compatible endpoint."""
    try:
        import httpx

        # Prepare headers for provider-specific models
        headers = {"Content-Type": "application/json"}

        # Add provider data headers if needed
        provider_headers = _get_provider_headers(model, None)
        headers.update(provider_headers)

        # Quick test request to see if the model is routable
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{base_url}/v1/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
            )

            # Check for various types of failures
            if response.status_code == 400:
                error_text = response.text.lower()
                if ("provider" in error_text and "not found" in error_text) or (
                    "invalid value" in error_text and "provider" in error_text
                ):
                    logger.debug(
                        f"Model {model} failed accessibility test:\
                         provider not configured"
                    )
                    return False

            # Check for 500 errors (LlamaStack internal server errors)
            elif response.status_code == 500:
                error_text = response.text.lower()
                if "usage" in error_text or "openai" in error_text:
                    logger.debug(
                        f"Model {model} failed accessibility\
                         test: LlamaStack internal error (likely usage attribute bug)"
                    )
                    return False

            # 200 or other errors suggest the model is
            # routable (even if it fails for other reasons)
            if response.status_code == 200:
                logger.debug(f"Model {model} passed accessibility test")
                return True
            else:
                # Some other error, but model is likely routable
                logger.debug(
                    f"Model {model} passed accessibility test\
                     (status: {response.status_code})"
                )
                return True

    except Exception as e:
        logger.debug(f"Model {model} accessibility test failed: {e}")
        return False


# Convenience functions for common operations
def get_llamastack_models(base_url: str = "http://localhost:8321") -> list[str]:
    """Get list of available models from LlamaStack.

    Args:
        base_url: LlamaStack base URL.

    Returns:
        List of available model names.

    Example:
        models = get_llamastack_models()
        print(f"Available models: {models}")
    """
    return list_available_models(base_url, model_type="llm")


def check_llamastack_status(base_url: str = "http://localhost:8321") -> dict[str, Any]:
    """Check LlamaStack connection and get status information.

    Args:
        base_url: LlamaStack base URL.

    Returns:
        Dictionary with connection status and model information.

    Example:
        status = check_llamastack_status()
        if status['connected']:
            print(f"LlamaStack is running with {status['models_count']} models")
        else:
            print(f"LlamaStack error: {status['error']}")
    """
    return check_llamastack_connection(base_url, model_type="llm")
