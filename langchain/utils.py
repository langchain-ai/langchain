"""Generic utility functions."""
import os
import re
from typing import Any, Callable, Dict, Literal, Optional, Tuple
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
    """
    The `render_prompt_and_examples` function generates a list of messages for a chatbot prompt, including a system message, human input message, AI output message, and optionally additional example pairs. The function takes the following parameters:

    - `system_prompt`: A string representing the initial system prompt to be displayed.
    - `input_prompt_template`: An f-string template representing the template for the human input prompt. The template should contain placeholder fields that will be replaced by the values in the `example` and `input_obj` dictionaries.
    - `output_prompt_template`: An f-string template representing the template for the AI output prompt. The template should contain placeholder fields that will be replaced by the values in `example` dictionaries.
    - `input_obj`: A dictionary containing input values for the prompt. These values will replace the placeholder fields in the final `input_prompt_template` (which should encourage the llm to generate the desired output prompt)
    - `examples`: An optional list of example dictionaries. Each dictionary represents an input-output example that will be displayed in the chatbot prompt.

    The function returns a list of `BaseMessage` objects, which represent the chatbot prompt messages.

    Examples:
    # These have not been tested yet
    ```
    # Example 1: Simple prompt with no examples
    input_obj = {"name": "Alice", "age": 25}
    messages = render_prompt_and_examples(
        "Welcome to the chatbot! What can I help you with today?",
        "Hello, my name is {name} and I am {age} years old.",
        "Nice to meet you, {name}!",
        input_obj
    )
    # Output:
    # [
    #     SystemMessage("Welcome to the chatbot! What can I help you with today?"),
    #     HumanMessage("Hello, my name is Alice and I am 25 years old."),
    #     AIMessage("Nice to meet you, Alice!"),
    #     HumanMessage("Hello, my name is {name} and I am {age} years old."),
    # ]

    # Example 2: Prompt with examples
    input_obj = {"name": "Bob", "age": 30}
    examples = [
        {"name": "John", "location": "New York"},
        {"name": "Jane", "location": "San Francisco"},
    ]
    messages = render_prompt_and_examples(
        "Welcome to the chatbot! What can I help you with today?",
        "Hello, my name is {name} and I am {age} years old. Can you tell me about {location}?",
        "Sure, {name} lives in {location}.",
        input_obj,
        examples=examples
    )
    # Output:
    # [
    #     SystemMessage("Welcome to the chatbot! What can I help you with today?"),
    #     HumanMessage("Hello, my name is Bob and I am 30 years old. Can you tell me about John?"),
    #     AIMessage("Sure, John lives in New York."),
    #     HumanMessage("Hello, my name is Bob and I am 30 years old. Can you tell me about Jane?"),
    #     AIMessage("Sure, Jane lives in San Francisco."),
    #     HumanMessage("Hello, my name is {name} and I am {age} years old. Can you tell me about {location}?"),
    # ]
    ```
    """
    # In case you're wondering why I'm converting everything into messages
    # instead of promopts, it's because messages are much more atomic and
    # easy to hash to cache. This leads me to believe that messages are the
    # fundamental unit of caching for most LLMAAS providers and message-based
    # prompting will become the cheapest way to generate text.
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


def infer_template_format(template: str) -> Literal["f-string", "jinja"]:
    """
    Infer the format of string template used in the given string.

    Args:
        template: The string template to analyze.

    Returns:
        A literal type annotation indicating the format of the template: either 'f-string' or 'jinja'.
    """
    if "{" in template and "}" in template:
        # The template contains curly braces, indicating an f-string
        return "f-string"
    elif "{%" in template or "{{" in template:
        # The template contains Jinja-specific syntax, indicating a Jinja template
        return "jinja"
    else:
        # The template does not match either type, default to f-string
        return "f-string"


def extract_input_variables(template: str, template_format: str) -> list[str]:
    """
    Extract the input variables used in the given string template.

    Args:
        template: The string template to analyze.
        template_type: The type of template used, either 'f-string' or 'jinja'.

    Returns:
        A list of unique input variable names used in the template, excluding local variables created inside loops.
    """

    if template_format == "f-string":
        pattern = r"{(.*?)}"
        matches = re.findall(pattern, template)
    elif template_format == "jinja":
        # Extract variables enclosed in double curly braces
        pattern = r"{{(.*?)}}"
        matches = re.findall(pattern, template)

        # Extract variables used in for loops (excluding loop variables)
        for_loop_pattern = r"{% for (.*?) in (.*?) %}"
        for_loop_matches = re.findall(for_loop_pattern, template)

        for loop_var, iterable_var in for_loop_matches:
            if iterable_var not in matches:
                matches.append(iterable_var)

    else:
        raise ValueError(f"Unsupported template type: {template_format}")

    # Remove loop variables
    loop_variables = (
        {match[0] for match in for_loop_matches} if template_format == "jinja" else set()
    )

    # Extract variable names, excluding those created in for loops
    input_variables = set()
    for match in matches:
        match = match.strip()
        if match not in loop_variables:
            input_variables.add(match)

    return sorted(list(input_variables))


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
