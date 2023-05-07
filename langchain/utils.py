"""Generic utility functions."""
import os
from typing import Any, Callable, Dict, Optional, Tuple
from typing import Any

from requests import HTTPError, Response

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


def xor_args(*arg_groups: Tuple[str, ...]) -> Callable:
    """Validate specified keyword args are mutually exclusive."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            """Validate exactly one arg in each group is not None."""
            counts = [
                sum(1 for arg in arg_group if kwargs.get(arg) is not None)
                for arg_group in arg_groups
            ]
            invalid_groups = [i for i, count in enumerate(counts) if count != 1]
            if invalid_groups:
                invalid_group_names = [", ".join(arg_groups[i]) for i in invalid_groups]
                raise ValueError(
                    "Exactly one argument in each of the following"
                    " groups must be defined:"
                    f" {', '.join(invalid_group_names)}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def raise_for_status_with_text(response: Response) -> None:
    """Raise an error with the response text."""
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e


def stringify_value(val: Any) -> str:
    if isinstance(val, str):
        return val
    elif isinstance(val, dict):
        return "\n" + stringify_dict(val)
    elif isinstance(val, list):
        return "\n".join(stringify_value(v) for v in val)
    else:
        return str(val)


def stringify_dict(data: dict) -> str:
    text = ""
    for key, value in data.items():
        text += key + ": " + stringify_value(value) + "\n"
    return text


def render_prompt_and_examples(
    system_prompt: str,
    input_prompt_template: str,
    output_prompt_template: str,
    input_obj: object,
    examples: list = [],
    input_keys: list[str] = [],
    output_keys: list[str] = [],
) -> list[BaseMessage]:
    return [
        SystemMessage(system_prompt),
        *sum(
            [
                HumanMessage(
                    input_prompt_template.format(
                        **{input_key: example[input_key] for input_key in input_keys}
                    )
                ),
                AIMessage(
                    output_prompt_template.format(
                        **{output_key: example[output_key] for output_key in output_keys}
                    )
                ),
            ]
            for example in examples
        ),
    ] + [
        input_prompt_template.format(
            **{input_key: input_obj[input_key] for input_key in input_keys}
        )
    ]


def comma_list(items: list[Any]) -> str:
    return ", ".join(str(item) for item in items)


def bullet_list(items: list[Any]) -> str:
    return "\n".join(f"- {str(item)}" for item in items)


def summarized_items(items: list[str], chars=30) -> str:
    return "\n\n".join(
        f"{str(item)[:chars]}..." if len(item) > chars else item for item in items
    )


def serialize_msgs(msgs: list[BaseMessage], include_type=False) -> str:
    return "\n\n".join(
        (f"{msg.type}: {msg.content}" if include_type else msg.content) for msg in msgs
    )
