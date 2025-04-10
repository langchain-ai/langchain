"""Utility functions for handling HTTP requests and responses."""

from requests import HTTPError, Response


def raise_for_status_with_text(response: Response) -> None:
    """Raise an error with the response text.

    Args:
        response (Response): The response to check for errors.

    Raises:
        ValueError: If the response has an error status code.
    """
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e
