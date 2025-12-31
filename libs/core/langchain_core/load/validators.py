"""Init validators for deserialization security.

This module contains extra validators that are called during deserialization, ex.
to prevent security issues such as SSRF attacks.

Each validator is a callable that takes a class path tuple and kwargs dict, and
raises an exception if the deserialization should be blocked.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.load.load import InitValidator


def _bedrock_validator(
    class_path: tuple[str, ...], kwargs: dict[str, Any]
) -> None:
    """Args input validator for AWS Bedrock integrations.

    Blocks deserialization if endpoint_url or base_url parameters are present, which
    could enable SSRF attacks.

    Args:
        class_path: The class path tuple being deserialized.
        kwargs: The kwargs dict for the class constructor.

    Raises:
        ValueError: If endpoint_url or base_url parameters are present.
    """
    dangerous_params = ["endpoint_url", "base_url"]
    found_params = [p for p in dangerous_params if p in kwargs]

    if found_params:
        class_name = class_path[-1] if class_path else "Unknown"
        param_str = " or ".join(found_params)
        msg = (
            f"Deserialization of {class_name} with {param_str} is not allowed "
            f"for security reasons. These parameters can enable Server-Side Request "
            f"Forgery (SSRF) attacks by directing network requests to arbitrary "
            f"endpoints during initialization. If you need to use a custom endpoint, "
            f"instantiate {class_name} directly rather than deserializing it."
        )
        raise ValueError(msg)


# Registry of class-specific validators, update as needed
CLASS_INIT_VALIDATORS: dict[tuple[str, ...], "InitValidator"] = {
    ("langchain", "chat_models", "bedrock", "ChatBedrock"): _bedrock_validator,
    ("langchain_aws", "chat_models", "ChatBedrockConverse"): _bedrock_validator,
    ("langchain", "llms", "bedrock", "Bedrock"): _bedrock_validator,
    ("langchain", "llms", "bedrock", "BedrockLLM"): _bedrock_validator,
}
