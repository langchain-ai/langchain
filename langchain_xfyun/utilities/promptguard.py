import json
from typing import Dict, Union


def sanitize(
    input: Union[str, Dict[str, str]]
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Sanitize input string or dict of strings by replacing sensitive data with
    placeholders.
    It returns the sanitized input string or dict of strings and the secure
    context as a dict following the format:
    {
        "sanitized_input": <sanitized input string or dict of strings>,
        "secure_context": <secure context>
    }

    The secure context is a bytes object that is needed to de-sanitize the response
    from the LLM.

    Args:
        input: Input string or dict of strings.

    Returns:
        Sanitized input string or dict of strings and the secure context
        as a dict following the format:
        {
            "sanitized_input": <sanitized input string or dict of strings>,
            "secure_context": <secure context>
        }

        The `secure_context` needs to be passed to the `desanitize` function.
    """
    try:
        import promptguard as pg
    except ImportError:
        raise ImportError(
            "Could not import the `promptguard` Python package, "
            "please install it with `pip install promptguard`."
        )

    if isinstance(input, str):
        # the input could be a string, so we sanitize the string
        sanitize_response: pg.SanitizeResponse = pg.sanitize(input)
        return {
            "sanitized_input": sanitize_response.sanitized_text,
            "secure_context": sanitize_response.secure_context,
        }

    if isinstance(input, dict):
        # the input could be a dict[string, string], so we sanitize the values
        values = list()

        # get the values from the dict
        for key in input:
            values.append(input[key])
        input_value_str = json.dumps(values)

        # sanitize the values
        sanitize_values_response: pg.SanitizeResponse = pg.sanitize(input_value_str)

        # reconstruct the dict with the sanitized values
        sanitized_input_values = json.loads(sanitize_values_response.sanitized_text)
        idx = 0
        sanitized_input = dict()
        for key in input:
            sanitized_input[key] = sanitized_input_values[idx]
            idx += 1

        return {
            "sanitized_input": sanitized_input,
            "secure_context": sanitize_values_response.secure_context,
        }

    raise ValueError(f"Unexpected input type {type(input)}")


def desanitize(sanitized_text: str, secure_context: bytes) -> str:
    """
    Restore the original sensitive data from the sanitized text.

    Args:
        sanitized_text: Sanitized text.
        secure_context: Secure context returned by the `sanitize` function.

    Returns:
        De-sanitized text.
    """
    try:
        import promptguard as pg
    except ImportError:
        raise ImportError(
            "Could not import the `promptguard` Python package, "
            "please install it with `pip install promptguard`."
        )
    desanitize_response: pg.DesanitizeResponse = pg.desanitize(
        sanitized_text, secure_context
    )
    return desanitize_response.desanitized_text
