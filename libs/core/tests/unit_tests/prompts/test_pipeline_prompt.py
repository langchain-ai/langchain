from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate


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


def test_final_prompt_input_variables() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("{bar}")
    prompt_final = PromptTemplate.from_template("{a}{b}{c}")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_final, pipeline_prompts=[("a", prompt_a), ("b", prompt_b)]
    )
    assert set(pipeline_prompt.input_variables) == set(["foo", "bar", "c"])

def test_overlapping_input_variables() -> None:
    prompt_a = PromptTemplate.from_template("{foo}withstr")
    prompt_b = PromptTemplate.from_template("{bar}")
    prompt_final = PromptTemplate.from_template("{a},{b},{c},{foo}")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_final, pipeline_prompts=[("a", prompt_a), ("b", prompt_b), ("foo", prompt_a)]
    )
    assert set(pipeline_prompt.input_variables) == set(["foo", "bar", "c"])
    assert pipeline_prompt.format(
        foo="FOO",
        bar="BAR",
        c="C"
    ) == "FOOwithstr,BAR,C,FOOwithstr"

# todo clean up
def test_overlapping_input_variables_2() -> None:
    user_prompt = PromptTemplate.from_template("Evaluate the following: {prompt}")
    system_prompt = PromptTemplate.from_template("You are a {adjective} robot.")
    prompt_final = PromptTemplate.from_template("<<SYS>> {system_prompt} <</SYS>> [INST] {prompt} [/INST]")
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=prompt_final, pipeline_prompts=[("system_prompt", system_prompt), ("prompt", user_prompt)]
    )
    assert set(pipeline_prompt.input_variables) == set(["prompt", "adjective"])
    assert pipeline_prompt.format(
        prompt="what is 2+2",
        adjective="kind",
    ) == "<<SYS>> You are a kind robot. <</SYS>> [INST] Evaluate the following: what is 2+2 [/INST]"
