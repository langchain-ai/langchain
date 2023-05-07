import string
from typing import Any

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


def render_prompt_and_examples(
    system_prompt: str,
    input_prompt_template: str,
    output_prompt_template: str,
    input_obj: dict[str, Any],
    examples: list[dict[str, Any]] = [],
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

    def _get_field_names(s):
        """
        Extract variable names from an f-string.

        Args:
            s (str): The f-string to extract variable names from.

        Returns:
            list[str]: The list of variable names in the f-string.
        """
        # Parse the f-string and extract the field names
        field_names = []
        for _, field_name, _, _ in string.Formatter().parse(s):
            if field_name is not None:
                field_names.append(field_name)

        return field_names

    input_keys = _get_field_names(input_prompt_template)
    output_keys = _get_field_names(output_prompt_template)

    return [
        *([SystemMessage(system_prompt)] if system_prompt else []),
        *sum(
            [
                HumanMessage(
                    input_prompt_template.format(
                        **{input_key: example[input_key] for input_key in input_keys}
                    )
                ),
                AIMessage(
                    output_prompt_template.format(
                        **{
                            output_key: example[output_key]
                            for output_key in output_keys
                        }
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
