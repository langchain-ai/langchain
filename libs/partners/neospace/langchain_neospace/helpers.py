from typing import Any, Mapping


def validate_extra_body(extra_body: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Validate the extra_body parameter.

    Args:
        extra_body (Mapping[str, Any]): The extra_body parameter to validate.

    Raises:
        ValueError: If the extra_body parameter does not have a 'session_id' field.

    """

    if 'session_id' not in extra_body:
        raise ValueError("extra_body parameter needs 'session_id' field.")
    
    return extra_body
