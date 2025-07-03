"""Utility functions for validating Ollama models."""

from httpx import ConnectError
from ollama import Client, ResponseError


def validate_model(client: Client, model_name: str) -> None:
    """Validate that a model exists in the Ollama instance.

    Args:
        client: The Ollama client.
        model_name: The name of the model to validate.

    Raises:
        ValueError: If the model is not found or if there's a connection issue.
    """
    try:
        response = client.list()
        model_names: list[str] = [model["name"] for model in response["models"]]
        if not any(
            model_name == m or m.startswith(f"{model_name}:") for m in model_names
        ):
            raise ValueError(
                f"Model `{model_name}` not found in Ollama. Please pull the "
                f"model (using `ollama pull {model_name}`) or specify a valid "
                f"model name. Available local models: {', '.join(model_names)}"
            )
    except ConnectError as e:
        raise ValueError(
            "Connection to Ollama failed. Please make sure Ollama is running "
            f"and accessible at {client._client.base_url}. "
        ) from e
    except ResponseError as e:
        raise ValueError(
            "Received an error from the Ollama API. "
            "Please check your Ollama server logs."
        ) from e
