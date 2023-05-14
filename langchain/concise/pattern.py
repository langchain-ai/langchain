from typing import TypeVar

from langchain.concise import config
from langchain.concise.generate import generate
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers.choice import ChoiceOutputParser
from langchain.schema import BaseOutputParser
from langchain.utils import render_prompt_and_examples

T = TypeVar("T")


def pattern(
    input: str,
    query: str = None,
    pattern_name: str = None,
    input_format_template: str = "{input}",
    output_format_template: str = "{output}",
    parser: BaseOutputParser[T] = None,
    examples: list[tuple[str, str]] = [],
    llm: BaseLanguageModel = None,
) -> T:
    llm = llm or config.get_default_llm()
    examples = [
        {"input": ex_input, "output": ex_output} for ex_input, ex_output in examples
    ]
    assert all(
        len(example) == 3 for example in examples
    ), "Examples must be a list of tuples of length 3."
    input_msgs = render_prompt_and_examples(
        system_prompt=query,
        input_prompt_template=input_format_template,
        output_prompt_template=output_format_template,
        input_obj={"input": input},
        examples=examples,
    )
    return generate(input_msgs, llm=llm, parser=parser)
