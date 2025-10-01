import pytest

from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

# Suppress deprecation warnings for PipelinePromptTemplate since we're testing the
# deprecated functionality intentionally to ensure it still works correctly


@pytest.mark.filterwarnings(
    "ignore:This class is deprecated"
    ":langchain_core._api.deprecation.LangChainDeprecationWarning"
)
def test_get_input_variables() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("{bar}")
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    assert pipeline_prompt.input_variables == ["foo"]


@pytest.mark.filterwarnings(
    "ignore:This class is deprecated"
    ":langchain_core._api.deprecation.LangChainDeprecationWarning"
)
def test_simple_pipeline() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("{bar}")
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    output = pipeline_prompt.format(foo="jim")
    assert output == "jim"


@pytest.mark.filterwarnings(
    "ignore:This class is deprecated"
    ":langchain_core._api.deprecation.LangChainDeprecationWarning"
)
def test_multi_variable_pipeline() -> None:
    prompt_a = PromptTemplate.from_template("{foo}")
    prompt_b = PromptTemplate.from_template("okay {bar} {baz}")
    pipeline_prompt = PipelinePromptTemplate(  # type: ignore[call-arg]
        final_prompt=prompt_b, pipeline_prompts=[("bar", prompt_a)]
    )
    output = pipeline_prompt.format(foo="jim", baz="deep")
    assert output == "okay jim deep"


@pytest.mark.filterwarnings(
    "ignore:This class is deprecated"
    ":langchain_core._api.deprecation.LangChainDeprecationWarning"
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
