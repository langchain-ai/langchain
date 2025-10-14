"""Utility function to validate Ollama models."""

from __future__ import annotations

import base64
from urllib.parse import unquote, urlparse

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


def parse_url_with_auth(
    url: str | None,
) -> tuple[str | None, dict[str, str] | None]:
    """Parse URL and extract `userinfo` credentials for headers.

    Handles URLs of the form: `https://user:password@host:port/path`

    Args:
        url: The URL to parse.

    Returns:
        A tuple of `(cleaned_url, headers_dict)` where:
        - `cleaned_url` is the URL without authentication credentials if any were
            found. Otherwise, returns the original URL.
        - `headers_dict` contains Authorization header if credentials were found.
    """
    if not url:
        return None, None

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        return None, None
    if not parsed.username:
        return url, None

    # Handle case where password might be empty string or None
    password = parsed.password or ""

    # Create basic auth header (decode percent-encoding)
    username = unquote(parsed.username)
    password = unquote(password)
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {"Authorization": f"Basic {encoded_credentials}"}

    # Strip credentials from URL
    cleaned_netloc = parsed.hostname or ""
    if parsed.port:
        cleaned_netloc += f":{parsed.port}"

    cleaned_url = f"{parsed.scheme}://{cleaned_netloc}"
    if parsed.path:
        cleaned_url += parsed.path
    if parsed.query:
        cleaned_url += f"?{parsed.query}"
    if parsed.fragment:
        cleaned_url += f"#{parsed.fragment}"

    return cleaned_url, headers


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
