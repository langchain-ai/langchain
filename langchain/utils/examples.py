from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


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
