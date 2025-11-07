#!/usr/bin/env python3
"""Refresh model profile data from models.dev.

Update the bundled model data by running:
    python scripts/refresh_data.py
"""

import json
from pathlib import Path

import httpx

PROVIDER_SUBSET = [
    # This is done to limit the data size
    "amazon-bedrock",
    "anthropic",
    "azure",
    "baseten",
    "cerebras",
    "cloudflare-workers-ai",
    "deepinfra",
    "deepseek",
    "fireworks-ai",
    "google",
    "google-vertex",
    "google-vertex-anthropic",
    "groq",
    "huggingface",
    "lmstudio",
    "mistral",
    "nebius",
    "nvidia",
    "openai",
    "openrouter",
    "perplexity",
    "togetherai",
    "upstage",
    "xai",
]


def main() -> None:
    """Download and save the latest model data from models.dev."""
    api_url = "https://models.dev/api.json"
    output_dir = Path(__file__).parent.parent / "langchain_model_profiles" / "data"
    output_file = output_dir / "models.json"

    print(f"Downloading data from {api_url}...")  # noqa: T201
    response = httpx.get(api_url, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Basic validation
    if not isinstance(data, dict):
        msg = "Expected API response to be a dictionary"
        raise TypeError(msg)

    provider_count = len(data)
    model_count = sum(len(provider.get("models", {})) for provider in data.values())

    print(f"Downloaded {provider_count} providers with {model_count} models")  # noqa: T201

    # Subset providers
    data = {k: v for k, v in data.items() if k in PROVIDER_SUBSET}
    print(f"Filtered to {len(data)} providers based on subset")  # noqa: T201

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write with pretty formatting for readability
    print(f"Writing to {output_file}...")  # noqa: T201
    with output_file.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"✓ Successfully refreshed model data ({output_file.stat().st_size:,} bytes)")  # noqa: T201


if __name__ == "__main__":
    main()
