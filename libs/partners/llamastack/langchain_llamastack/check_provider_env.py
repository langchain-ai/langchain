#!/usr/bin/env python3
"""
Simple Provider Environment Variable Check
==========================================

Shows which LlamaStack providers have their environment variables set.
Uses hardcoded provider mapping - no API calls required.
"""

import os


def check_provider_environment_variables() -> bool:
    """Simple check of provider environment variables."""

    # Standard mapping based on common LlamaStack provider patterns
    providers = {
        "ollama": {
            "env_var": "OLLAMA_URL",
            "description": "Ollama local inference server",
            "default": "http://localhost:11434",
        },
        "together": {
            "env_var": "TOGETHER_API_KEY",
            "description": "Together AI cloud inference",
            "default": None,
        },
        "fireworks": {
            "env_var": "FIREWORKS_API_KEY",
            "description": "Fireworks AI cloud inference",
            "default": None,
        },
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "description": "OpenAI cloud inference",
            "default": None,
        },
        "anthropic": {
            "env_var": "ANTHROPIC_API_KEY",
            "description": "Anthropic Claude cloud inference",
            "default": None,
        },
        "vllm": {
            "env_var": "VLLM_URL",
            "description": "vLLM inference server",
            "default": None,
        },
        "meta-reference": {
            "env_var": "META_REFERENCE_API_KEY",
            "description": "Meta reference implementation",
            "default": None,
        },
        "bedrock": {
            "env_var": "AWS_ACCESS_KEY_ID",
            "description": "AWS Bedrock cloud inference",
            "default": None,
        },
        "vertex": {
            "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
            "description": "Google Vertex AI cloud inference",
            "default": None,
        },
    }

    configured_count = 0

    for provider_id, config in providers.items():
        env_var = config["env_var"]
        env_value = os.getenv(env_var)

        # Special handling for OLLAMA_URL (has default)
        if provider_id == "ollama" and not env_value:
            env_value = config["default"]
        elif env_value:
            # Mask API keys
            if "api_key" in env_var.lower() or "key" in env_var.lower():
                env_value = "***"
            else:
                pass
        else:
            pass

        if env_value:
            configured_count += 1

    if configured_count == 0:
        pass

    return configured_count > 0


if __name__ == "__main__":
    import sys

    success = check_provider_environment_variables()
    sys.exit(0 if success else 1)
