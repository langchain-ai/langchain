import pytest

from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.prompts.pipeline import PipelinePromptTemplate


def test_pipeline_prompt_template_basic_formatting() -> None:
    full_template = """{introduction}

{example}

{start}"""
    full_prompt = PromptTemplate.from_template(full_template)

    intro_template = """Here's a roleplay with {person}."""
    intro_prompt = PromptTemplate.from_template(intro_template)

    example_template = """Q: {example_q}
A: {example_a}"""
    example_prompt = PromptTemplate.from_template(example_template)

    start_template = """Now answer this: {input}"""
    start_prompt = PromptTemplate.from_template(start_template)

    input_prompts = [
        ("introduction", intro_prompt),
        ("example", example_prompt),
        ("start", start_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts
    )

    output = pipeline_prompt.format(
        person="Elon",
        example_q="What's Mars?",
        example_a="A planet.",
        input="What is SpaceX?",
    )

    expected = """Here's a roleplay with Elon.

Q: What's Mars?
A: A planet.

Now answer this: What is SpaceX?"""
    assert output == expected


def test_pipeline_prompt_template_nested_pipeline() -> None:
    outer_template = """Start: {inner}
End."""
    outer_prompt = PromptTemplate.from_template(outer_template)

    inner_template = """Inner: {msg}"""
    inner_prompt = PromptTemplate.from_template(inner_template)

    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=outer_prompt,
        pipeline_prompts=[("inner", inner_prompt)],
    )

    res = pipeline_prompt.format_prompt(msg="Hello Nested").to_string()
    assert res == "Start: Inner: Hello Nested\nEnd."


def test_pipeline_prompt_template_input_variables() -> None:
    full_template = """{header} -> {content}"""
    full_prompt = PromptTemplate.from_template(full_template)

    header_prompt = PromptTemplate.from_template("Header: {title}")
    content_prompt = PromptTemplate.from_template("Body: {body_text}")

    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt,
        pipeline_prompts=[
            ("header", header_prompt),
            ("content", content_prompt),
        ],
    )

    assert set(pipeline_prompt.input_variables) == {"title", "body_text"}
