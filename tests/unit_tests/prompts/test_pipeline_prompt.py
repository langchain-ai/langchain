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
