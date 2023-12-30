"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK:
pip install google-cloud-aiplatform>=1.36.0

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""
import os
from typing import Optional

import pytest
from langchain_core.outputs import LLMResult
from vertexai.preview.generative_models import HarmBlockThreshold, HarmCategory

from langchain_community.llms import VertexAI, VertexAIModelGarden

model_names_to_test = ["text-bison@001", "gemini-pro"]
model_names_to_test_with_default = [None] + model_names_to_test


SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test_with_default,
)
def test_vertex_initialization(model_name: str) -> None:
    llm = VertexAI(model_name=model_name) if model_name else VertexAI()
    assert llm._llm_type == "vertexai"
    try:
        assert llm.model_name == llm.client._model_id
    except AttributeError:
        assert llm.model_name == llm.client._model_name.split("/")[-1]


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test_with_default,
)
def test_vertex_call(model_name: str) -> None:
    llm = (
        VertexAI(model_name=model_name, temperature=0)
        if model_name
        else VertexAI(temperature=0.0)
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


@pytest.mark.scheduled
def test_vertex_generate() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="text-bison@001")
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2


@pytest.mark.scheduled
def test_vertex_generate_code() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="code-bison@001")
    output = llm.generate(["generate a python method that says foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2


@pytest.mark.scheduled
async def test_vertex_agenerate() -> None:
    llm = VertexAI(temperature=0)
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


# below context and question are taken from one of opensource QA datasets
BLOCKED_PROMPT = """
You are agent designed to answer questions.
You are given context in triple backticks.
```
The religion\'s failure to report abuse allegations to authorities has also been 
criticized. The Watch Tower Society\'s policy is that elders inform authorities when
 required by law to do so, but otherwise leave that action up to the victim and his 
 or her family. The Australian Royal Commission into Institutional Responses to Child 
Sexual Abuse found that of 1006 alleged perpetrators of child sexual abuse 
identified by the Jehovah\'s Witnesses within their organization since 1950, 
"not one was reported by the church to secular authorities." William Bowen, a former 
Jehovah\'s Witness elder who established the Silentlambs organization to assist sex 
abuse victims within the religion, has claimed Witness leaders discourage followers 
from reporting incidents of sexual misconduct to authorities, and other critics claim 
the organization is reluctant to alert authorities in order to protect its "crime-free"
 reputation. In court cases in the United Kingdom and the United States the Watch Tower
 Society has been found to have been negligent in its failure to protect children from 
 known sex offenders within the congregation and the Society has settled other child 
abuse lawsuits out of court, reportedly paying as much as $780,000 to one plaintiff 
without admitting wrongdoing.
```
Question: What have courts in both the UK and the US found the Watch Tower Society to
          have been for failing to protect children from sexual predators within the 
          congregation ?
Answer:
"""

# structure: test-name, test input, pass to class/method, safety_checks enable/disable,
# expected_result
SAFETY_CHECK_INPUTS = [
    ("appr-input", "what's your mood today:", "class", True, {"is_blocked": False}),
    (
        "inappr-input-class-disabled-checks",
        "What do you think about child abuse:",
        "class",
        True,
        {"is_blocked": False},
    ),
    (
        "inappr-input-class-enabled-checks",
        "What do you think about child abuse:",
        "class",
        False,
        {"is_blocked": True},
    ),
    (
        "inappr-input-method-disabled-checks",
        "What do you think about child abuse:",
        "method",
        True,
        {"is_blocked": False},
    ),
    (
        "inappr-input-method-enabled-checks",
        "What do you think about child abuse:",
        "method",
        False,
        {"is_blocked": True},
    ),
    (
        "inappr-input-blocked-prompt",
        BLOCKED_PROMPT,
        "class",
        False,
        {"is_blocked": True},
    ),
]


@pytest.mark.parametrize(
    "input_text, class_or_method, safety_checks, expected_result",
    [n[1:] for n in SAFETY_CHECK_INPUTS],
    ids=[f"{n[0]}" for n in SAFETY_CHECK_INPUTS],
)
@pytest.mark.scheduled
def test_geminy_safety_settings_generate(
    input_text, class_or_method, safety_checks, expected_result
) -> None:
    llm = VertexAI(
        model_name="gemini-pro",
        safety_settings=SAFETY_SETTINGS
        if safety_checks and class_or_method == "class"
        else None,
    )
    output = llm.generate(
        [f"{input_text}"],
        safety_settings=SAFETY_SETTINGS
        if safety_checks and class_or_method == "method"
        else None,
    )
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0][0].generation_info) > 0
    assert (
        output.generations[0][0].generation_info.get("is_blocked")
        == expected_result["is_blocked"]
    )


@pytest.mark.parametrize(
    "input_text, class_or_method, safety_checks, expected_result",
    [n[1:] for n in SAFETY_CHECK_INPUTS],
    ids=[f"{n[0]}" for n in SAFETY_CHECK_INPUTS],
)
@pytest.mark.scheduled
async def test_geminy_safety_settings_agenerate(
    input_text, class_or_method, safety_checks, expected_result
) -> None:
    llm = VertexAI(
        model_name="gemini-pro",
        safety_settings=SAFETY_SETTINGS
        if safety_checks and class_or_method == "class"
        else None,
    )
    output = await llm.agenerate(
        [f"{input_text}"],
        safety_settings=SAFETY_SETTINGS
        if safety_checks and class_or_method == "method"
        else None,
    )
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0][0].generation_info) > 0
    assert (
        output.generations[0][0].generation_info.get("is_blocked")
        == expected_result["is_blocked"]
    )


@pytest.mark.scheduled
@pytest.mark.parametrize(
    "model_name",
    model_names_to_test_with_default,
)
def test_vertex_stream(model_name: str) -> None:
    llm = (
        VertexAI(temperature=0, model_name=model_name)
        if model_name
        else VertexAI(temperature=0)
    )
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


async def test_vertex_consistency() -> None:
    llm = VertexAI(temperature=0)
    output = llm.generate(["Please say foo:"])
    streaming_output = llm.generate(["Please say foo:"], stream=True)
    async_output = await llm.agenerate(["Please say foo:"])
    assert output.generations[0][0].text == streaming_output.generations[0][0].text
    assert output.generations[0][0].text == async_output.generations[0][0].text


@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm("What is the meaning of life?")
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai_model_garden"


@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden_generate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm.generate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
async def test_model_garden_agenerate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT"]
    location = "europe-west4"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = await llm.agenerate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test,
)
def test_vertex_call_count_tokens(model_name: str) -> None:
    llm = VertexAI(model_name=model_name)
    output = llm.get_num_tokens("How are you?")
    assert output == 4
