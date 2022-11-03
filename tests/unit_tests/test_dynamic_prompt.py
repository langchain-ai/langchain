"""Test functionality related to dynamic prompts."""
from langchain.prompt import DynamicPrompt, Prompt


def test_dynamic_prompt_valid() -> None:
    """Test dynamic prompt can be successfully constructed from examples."""
    template = """Test Prompt:

Question: who are you?
Answer: foo

Question: what are you?
Answer: bar

Question: {question}
Answer:"""
    input_variables = ["question"]
    example_separator = "\n\n"
    prefix = """Test Prompt:\n\n"""
    suffix = """\n\nQuestion: {question}\nAnswer:"""
    examples = [
        """Question: who are you?\nAnswer: foo""",
        """Question: what are you?\nAnswer: bar""",
    ]
    dynamic_prompt = DynamicPrompt(
        examples=examples,
        suffix=suffix,
        input_variables=input_variables,
        example_separator=example_separator,
        prefix=prefix,
    )
    prompt_from_template = Prompt(input_variables=input_variables, template=template)
    assert dynamic_prompt.format(question="foo?") == prompt_from_template.format(
        question="foo?"
    )
    assert dynamic_prompt.input_variables == prompt_from_template.input_variables


def test_dynamic_prompt_trims_examples() -> None:
    """Test dynamic prompt can be successfully constructed from examples."""
    longer_template = """Test Prompt:

Question: who are you?
Answer: foo

Question: what are you?
Answer: bar

Question: {question}
Answer:"""
    shorter_template = """Test Prompt:

Question: who are you?
Answer: foo

Question: {question}
Answer:"""
    input_variables = ["question"]
    example_separator = "\n\n"
    prefix = """Test Prompt:\n\n"""
    suffix = """\n\nQuestion: {question}\nAnswer:"""
    examples = [
        """Question: who are you?\nAnswer: foo""",
        """Question: what are you?\nAnswer: bar""",
    ]
    dynamic_prompt = DynamicPrompt(
        examples=examples,
        suffix=suffix,
        input_variables=input_variables,
        example_separator=example_separator,
        prefix=prefix,
        max_length=30,
    )
    test_long_question = """I am writing a really long question,
this probably is going to affect the example right?"""
    test_short_question = "Short question?"

    assert dynamic_prompt.format(
        question=test_long_question
    ) == shorter_template.format(question=test_long_question)
    assert dynamic_prompt.format(
        question=test_short_question
    ) == longer_template.format(question=test_short_question)
