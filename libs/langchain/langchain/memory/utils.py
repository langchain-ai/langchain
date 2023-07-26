from typing import Any, Dict, List

from langchain.schema.messages import get_buffer_string  # noqa: 401


def get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    """
    Get the prompt input key.

    Args:
        inputs: Dict[str, Any]
        memory_variables: List[str]

    Returns:
        A prompt input key.
    """
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]
