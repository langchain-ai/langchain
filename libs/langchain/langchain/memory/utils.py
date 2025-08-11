from typing import Any


def get_prompt_input_key(inputs: dict[str, Any], memory_variables: list[str]) -> str:
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
    prompt_input_keys = list(set(inputs).difference([*memory_variables, "stop"]))
    if len(prompt_input_keys) != 1:
        msg = f"One input key expected got {prompt_input_keys}"
        raise ValueError(msg)
    return prompt_input_keys[0]
