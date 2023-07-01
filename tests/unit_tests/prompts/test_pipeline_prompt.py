from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate


def test_get_input_variables() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("{bar}")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    assert pipeline_prompt.input_variables == ["foo"]


def test_simple_pipeline() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("{bar}")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    output = pipeline_prompt.format(foo="jim")
    assert output == "jim"


def test_multi_variable_pipeline() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("okay {bar} {baz}")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    output = pipeline_prompt.format(foo="jim", baz="deep")
    assert output == "okay jim deep"


def test_partial_with_chat_prompts() -> None:
    prompt_a = ChatPromptTemplate(
        input_variables=["foo"], messages=[MessagesPlaceholder(variable_name="foo")]
    )
    prompt_b = ChatPromptTemplate.from_template("jim {bar}")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_a, pipeline_prompts=[("foo", prompt_b)]
    )
    assert pipeline_prompt.input_variables == ["bar"]
    output = pipeline_prompt.format_prompt(bar="okay")
    assert output.to_messages()[0].content == "jim okay"


def test_missing_variable() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("okay {bar}")
    prompt_c = PromptTemplate.from_template("{A} {B}")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_c, pipeline_prompts=[("A", prompt_a), ("B", prompt_b)]
    )
    output = pipeline_prompt.format(foo="jim")

    print(output)
    assert output == "jim okay "


def test_missing_variable_jinja2() -> None:
    prompt_a = PromptTemplate.from_template("{{foo}}", template_format="jinja2")
    prompt_b = PromptTemplate.from_template("okay {{bar}}", template_format="jinja2")
    prompt_c = PromptTemplate.from_template("{{A}} {{B}}", template_format="jinja2")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_c,
        pipeline_prompts=[("A", prompt_a), ("B", prompt_b)],
    )
    output = pipeline_prompt.format(foo="jim")

    print(output)
    assert output == "jim okay "
