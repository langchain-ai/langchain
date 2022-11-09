"""Test functionality related to dynamic prompts."""
from langchain.prompts.dynamic import DynamicPrompt
from langchain.prompts.prompt import Prompt

# FULL TEMPLATES
LONGER_TEMPLATE = """Test Prompt:

Question: who are you?
Answer: foo

Question: what are you?
Answer: bar

Question: {question}
Answer:"""
SHORTER_TEMPLATE = """Test Prompt:

Question: who are you?
Answer: foo

Question: {question}
Answer:"""
SHORTEST_TEMPLATE = """Test Prompt:

Question: {question}
Answer:"""

# DYNAMIC PROMPT COMPONENTS
PREFIX = """Test Prompt:"""
SUFFIX = """Question: {question}\nAnswer:"""
EXAMPLES = [
    """Question: who are you?\nAnswer: foo""",
    """Question: what are you?\nAnswer: bar""",
]

# INPUTS
TEST_LONG_QUESTION = """I am writing a really long question,
this probably is going to affect the example right?"""
TEST_LONGEST_QUESTION = """This question is super super super,
super super super super super super super super super super super,
super super super super long, this will affect the example right?"""
TEST_SHORT_QUESTION = "Short question?"


def test_dynamic_prompt_valid() -> None:
    """Test dynamic prompt can be successfully constructed from examples."""
    input_variables = ["question"]
    example_separator = "\n\n"
    dynamic_prompt_cls = DynamicPrompt(
        examples=EXAMPLES,
        suffix=SUFFIX,
        input_variables=input_variables,
        example_separator=example_separator,
        prefix=PREFIX,
    )
    prompt_cls = Prompt(input_variables=input_variables, template=LONGER_TEMPLATE)
    dynamic_prompt_template = dynamic_prompt_cls.format(question="foo?")
    prompt_template = prompt_cls.format(question="foo?")
    assert dynamic_prompt_template == prompt_template
    assert dynamic_prompt_cls.input_variables == prompt_cls.input_variables


def test_dynamic_prompt_trims_one_example() -> None:
    """Test dynamic prompt can trim one example."""
    input_variables = ["question"]
    example_separator = "\n\n"
    dynamic_prompt_cls = DynamicPrompt(
        examples=EXAMPLES,
        suffix=SUFFIX,
        input_variables=input_variables,
        example_separator=example_separator,
        prefix=PREFIX,
        max_length=30,
    )
    dynamic_prompt = dynamic_prompt_cls.format(question=TEST_LONG_QUESTION)
    shorter_prompt = SHORTER_TEMPLATE.format(question=TEST_LONG_QUESTION)
    assert dynamic_prompt == shorter_prompt


def test_dynamic_prompt_trims_no_examples() -> None:
    """Test dynamic prompt can trim no examples."""
    input_variables = ["question"]
    example_separator = "\n\n"
    dynamic_prompt_cls = DynamicPrompt(
        examples=EXAMPLES,
        suffix=SUFFIX,
        input_variables=input_variables,
        example_separator=example_separator,
        prefix=PREFIX,
        max_length=30,
    )
    dynamic_prompt = dynamic_prompt_cls.format(question=TEST_SHORT_QUESTION)
    full_prompt = LONGER_TEMPLATE.format(question=TEST_SHORT_QUESTION)
    assert dynamic_prompt == full_prompt


def test_dynamic_prompt_trims_all_examples() -> None:
    """Test dynamic prompt can trim all examples."""
    input_variables = ["question"]
    example_separator = "\n\n"
    dynamic_prompt_cls = DynamicPrompt(
        examples=EXAMPLES,
        suffix=SUFFIX,
        input_variables=input_variables,
        example_separator=example_separator,
        prefix=PREFIX,
        max_length=30,
    )
    dynamic_prompt = dynamic_prompt_cls.format(question=TEST_LONGEST_QUESTION)
    full_prompt = SHORTEST_TEMPLATE.format(question=TEST_LONGEST_QUESTION)
    assert dynamic_prompt == full_prompt
