from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate


def test_get_input_variables() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("{bar}")
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    assert pipeline_prompt.input_variables == ["foo"]


def test_simple_pipeline() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("{bar}")
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    output = pipeline_prompt.format(foo="jim")
    assert output == "jim"


def test_multi_variable_pipeline() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("okay {bar} {baz}")
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    output = pipeline_prompt.format(foo="jim", baz="deep")
    assert output == "okay jim deep"


def test_partial_with_prompt_template() -> None:
    full_template = """{introduction}
      {example}
      {start}"""
    full_prompt = PromptTemplate.from_template(full_template)

    introduction_template = """You are impersonating {person}."""
    introduction_prompt = PromptTemplate.from_template(introduction_template)

    example_template = """Here's an example of an interaction:
    Q: {example_q}
    A: {example_a}"""
    example_prompt = PromptTemplate.from_template(example_template)

    start_template = """Now, do this for real!
    Q: {input}
    A:"""
    start_prompt = PromptTemplate.from_template(start_template)

    input_prompts = [
        ("introduction", introduction_prompt),
        ("example", example_prompt),
        ("start", start_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=full_prompt,
        pipeline_prompts=input_prompts,  # type: ignore[arg-type]
    )

    pipeline_prompt.partial(person="Elon Musk")
    pipeline_prompt.partial(invalid_partial="Hello, I am invalid")

    ret = pipeline_prompt.format(
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
    assert (
        ret
        == """You are impersonating Elon Musk.
      Here's an example of an interaction:
    Q: What's your favorite car?
    A: Tesla
      Now, do this for real!
    Q: What's your favorite social media site?
    A:"""
    )


async def test_partial_with_chat_prompts() -> None:
    prompt_a = ChatPromptTemplate(
        input_variables=["foo"], messages=[MessagesPlaceholder(variable_name="foo")]
    )
    prompt_b = ChatPromptTemplate.from_template("jim {bar}")
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=prompt_a, pipeline_prompts=[("foo", prompt_b)]
    )
    assert pipeline_prompt.input_variables == ["bar"]
    output = pipeline_prompt.format_prompt(bar="okay")
    assert output.to_messages()[0].content == "jim okay"
    output = await pipeline_prompt.aformat_prompt(bar="okay")
    assert output.to_messages()[0].content == "jim okay"
