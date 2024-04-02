from __future__ import annotations

import logging
import re
from typing import Any, Callable

import cohere
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

JSON_TO_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "array": "List",
    "object": "Dict",
}


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    # support v4 and v5
    retry_conditions = (
        retry_if_exception_type(cohere.error.CohereError)
        if hasattr(cohere, "error")
        else retry_if_exception_type(Exception)
    )

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_conditions,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _remove_signature_from_tool_description(name: str, description: str) -> str:
    """
    Removes the `{name}{signature} - ` prefix and Args: section from tool description.
    The signature is usually present for tools created with the @tool decorator,
    whereas the Args: section may be present in function doc blocks.
    """
    description = re.sub(rf"^{name}\(.*?\) -(?:> \w+? -)? ", "", description)
    description = re.sub(r"(?s)(?:\n?\n\s*?)?Args:.*$", "", description)
    return description
