"""Test BigFrames Large Language Models
In order to run this test, you need to have an account on Google Cloud.

pip install bigframes
"""

from langchain.llms.bigframesllm import BigFramesLLM
from langchain import LLMChain,PromptTemplate
import bigframes.pandas as bf
import pytest

TEST_CONNECTION = "bigframes-dev.us.bigframes-ml"

@pytest.fixture
def bigframes_session():
    bf.options.bigquery.project = "bigframes-dev"
    bf.options.bigquery.location = "US"
    session = bf.get_global_session()
    yield session
    session.close()


def test_bigframesllm_initialization(bigframes_session) -> None:
    llm = BigFramesLLM(session=bigframes_session, connection=TEST_CONNECTION)
    assert llm._llm_type == "bigframesllm"
    assert llm.model_name == "PaLM2TextGenerator"
    output = llm(
        "What is the capital of France ?"
    )
    assert output == " The capital of France is Paris."


def test_bigframesllm_chained(bigframes_session) -> None:
    """Test valid call to bigframesllm."""
    llm = BigFramesLLM(session=bigframes_session, connection=TEST_CONNECTION)
    template = """Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    # answer is a bigframes dataframe
    answer = llm_chain.run("What is BigFrames?")
    assert answer.startswith(" BigFrames is a distributed computing framework for processing massive data sets.")
