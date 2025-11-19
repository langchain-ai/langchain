"""CLI for refreshing model profile data from models.dev."""

import argparse
import json
import sys
from pathlib import Path

import httpx


def refresh(provider: str, output: Path) -> None:
    """Download and save model data from models.dev for a specific provider.

    Args:
        provider: Provider ID from models.dev (e.g., 'anthropic', 'openai').
        output: Output path for the models.json file.
    """
    api_url = "https://models.dev/api.json"

    print(f"Provider: {provider}")  # noqa: T201
    print(f"Output path: {output}")  # noqa: T201
    print()  # noqa: T201

    # Download data from models.dev
    print(f"Downloading data from {api_url}...")  # noqa: T201
    response = httpx.get(api_url, timeout=30)
    response.raise_for_status()

    all_data = response.json()

    # Basic validation
    if not isinstance(all_data, dict):
        msg = "Expected API response to be a dictionary"
        raise TypeError(msg)

    provider_count = len(all_data)
    model_count = sum(len(p.get("models", {})) for p in all_data.values())
    print(f"Downloaded {provider_count} providers with {model_count} models")  # noqa: T201

    # Extract data for this provider
    if provider not in all_data:
        msg = f"Provider '{provider}' not found in models.dev data"
        print(msg, file=sys.stderr)  # noqa: T201
        sys.exit(1)

    provider_data = {provider: all_data[provider]}
    provider_model_count = len(provider_data[provider].get("models", {}))
    print(f"Extracted {provider_model_count} models for {provider}")  # noqa: T201

    # Ensure directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write with pretty formatting for readability
    print(f"Writing to {output}...")  # noqa: T201
    with output.open("w") as f:
        json.dump(provider_data, f, indent=2, sort_keys=True)

    print(f"✓ Successfully refreshed model data ({output.stat().st_size:,} bytes)")  # noqa: T201


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Refresh model profile data from models.dev",
        prog="langchain-profiles",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # refresh command
    refresh_parser = subparsers.add_parser(
        "refresh", help="Download and save model data for a provider"
    )
    refresh_parser.add_argument(
        "--provider",
        required=True,
        help="Provider ID from models.dev (e.g., 'anthropic', 'openai', 'google')",
    )
    refresh_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for models.json file",
    )

    args = parser.parse_args()

    if args.command == "refresh":
        refresh(args.provider, args.output)


if __name__ == "__main__":
    main()
