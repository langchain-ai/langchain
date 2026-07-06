"""Utility function to validate Ollama models."""

from __future__ import annotations

import base64
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

from httpx import ConnectError
from ollama import Client, ResponseError


def validate_model(client: Client, model_name: str) -> None:
    """Validate that a model exists in the local Ollama instance.

    Args:
        client: The Ollama client.
        model_name: The name of the model to validate.

    Raises:
        ValueError: If the model is not found or if there's a connection issue.
    """
    try:
        response = client.list()

        model_names: list[str] = [model["model"] for model in response["models"]]

        if not any(
            model_name == m or m.startswith(f"{model_name}:") for m in model_names
        ):
            msg = (
                f"Model `{model_name}` not found in Ollama. Please pull the "
                f"model (using `ollama pull {model_name}`) or specify a valid "
                f"model name. Available local models: {', '.join(model_names)}"
            )
            raise ValueError(msg)
    except ConnectError as e:
        msg = (
            "Failed to connect to Ollama. Please check that Ollama is downloaded, "
            "running and accessible. https://ollama.com/download"
        )
        raise ValueError(msg) from e
    except ResponseError as e:
        msg = (
            "Received an error from the Ollama API. "
            "Please check your Ollama server logs."
        )
        raise ValueError(msg) from e


def _build_cleaned_url(parsed: ParseResult) -> str:
    """Reconstruct a URL from parsed components without userinfo.

    Args:
        parsed: Parsed URL components.

    Returns:
        Cleaned URL string with userinfo removed.
    """
    hostname = parsed.hostname or ""
    if ":" in hostname:  # IPv6 — re-add brackets stripped by urlparse
        hostname = f"[{hostname}]"
    cleaned_netloc = hostname
    if parsed.port is not None:
        cleaned_netloc += f":{parsed.port}"
    return urlunparse(
        (
            parsed.scheme,
            cleaned_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def parse_url_with_auth(
    url: str | None,
) -> tuple[str | None, dict[str, str] | None]:
    """Parse URL and extract `userinfo` credentials for headers.

    Handles URLs of the form: `https://user:password@host:port/path`

    Scheme-less URLs (e.g., `host:port`) are also accepted and will be
    given a default `http://` scheme.

    Args:
        url: The URL to parse.

    Returns:
        A tuple of `(cleaned_url, headers_dict)` where:
        - `cleaned_url` is a normalized URL with credentials stripped (if any
            were present) and a scheme guaranteed (defaulting to `http://` for
            scheme-less inputs). Returns the original URL unchanged when it
            already has a valid scheme and no credentials.
        - `headers_dict` contains Authorization header if credentials were found.
    """
    if not url:
        return None, None

    parsed = urlparse(url)
    needs_reconstruction = False
    valid = False

    if parsed.scheme in {"http", "https"} and parsed.netloc and parsed.hostname:
        valid = True
    elif not (parsed.scheme and parsed.netloc) and ":" in url:
        # No valid scheme but contains colon — try as scheme-less host:port
        parsed_with_scheme = urlparse(f"http://{url}")
        if parsed_with_scheme.netloc and parsed_with_scheme.hostname:
            parsed = parsed_with_scheme
            needs_reconstruction = True
            valid = True

    # Validate port is numeric (urlparse raises ValueError for non-numeric ports)
    if valid:
        try:
            _ = parsed.port
        except ValueError:
            valid = False

    if not valid:
        return None, None

    if not parsed.username:
        cleaned = _build_cleaned_url(parsed) if needs_reconstruction else url
        return cleaned, None

    # Handle case where password might be empty string or None
    password = parsed.password or ""

    # Create basic auth header (decode percent-encoding)
    username = unquote(parsed.username)
    password = unquote(password)
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {"Authorization": f"Basic {encoded_credentials}"}

    return _build_cleaned_url(parsed), headers


def merge_auth_headers(
    client_kwargs: dict,
    auth_headers: dict[str, str] | None,
) -> None:
    """Merge authentication headers into client kwargs in-place.

    Args:
        client_kwargs: The client kwargs dict to update.
        auth_headers: Headers to merge (typically from `parse_url_with_auth`).
    """
    if auth_headers:
        headers = client_kwargs.get("headers", {})
        headers.update(auth_headers)
        client_kwargs["headers"] = headers
