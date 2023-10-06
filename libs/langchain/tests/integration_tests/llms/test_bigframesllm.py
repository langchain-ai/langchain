"""Test BigFrames Large Language Models
In order to run this test, you need to have an account on Google Cloud.

pip install bigframes
"""

import bigframes.pandas as bf

from langchain import LLMChain, PromptTemplate
from langchain.chains import BigFramesChain
from langchain.llms.bigframesllm import BigFramesLLM

TEST_CONNECTION = "bigframes-dev.us.bigframes-ml"


# We don't make this a pytest fixture since bigframes session will expire.
def bigframes_session():
    bf.reset_session()
    bf.options.bigquery.project = "bigframes-dev"
    bf.options.bigquery.location = "US"
    session = bf.get_global_session()
    return session


def test_bigframesllm_initialization_str() -> None:
    session = bigframes_session()
    llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
    assert llm._llm_type == "bigframesllm"
    assert llm.model_name == "PaLM2TextGenerator"
    # output is a Bigframes DataFrame
    output = llm("What is the capital of France ?")
    assert "ml_generate_text_llm_result" in output.columns
    assert output["ml_generate_text_llm_result"][0] == " The capital of France is Paris."


def test_bigframesllm_initialization_df() -> None:
    session = bigframes_session()
    llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
    assert llm._llm_type == "bigframesllm"
    assert llm.model_name == "PaLM2TextGenerator"
    df = bf.DataFrame(
        {
            "prompt": [
                "What is BigQuery?",
                "What is BQML?",
                "What is BigQuery DataFrame?",
            ],
        }
    )
    # output is a Bigframes DataFrame
    output = llm(df)
    assert "ml_generate_text_llm_result" in output.columns
    series = output["ml_generate_text_llm_result"]
    assert series[0].startswith(
        " BigQuery is Google's fully managed, petabyte-scale analytics "
        + "data warehouse")


def test_bigframesllm_chained_run() -> None:
    """Test valid call to bigframesllm."""
    session = bigframes_session()
    llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
    template = """Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run("What is BigFrames?")
    assert answer.startswith(
        " BigFrames is a distributed computing framework"
        + " for processing massive data sets."
    )


def test_bigframesllm_chained_invoke() -> None:
    """Test valid call to bigframesllm."""
    session = bigframes_session()
    llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
    template = """Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # answer is a string
    answer = llm_chain.invoke({"question":"What is BigFrames?"})
    print(answer)
    assert answer["question"] == "What is BigFrames?"
    assert answer["text"].startswith(
        " BigFrames is a distributed computing framework"
        + " for processing massive data sets."
    )


def test_bigframeschain_chained_call() -> None:
    """Test valid call to bigframesllm."""
    session = bigframes_session()
    llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
    template = """Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = BigFramesChain(prompt=prompt, llm=llm)

    # answer is a BigFrames DataFrame
    answer = llm_chain("What is BigFrames?")
    print(answer)
    assert "ml_generate_text_llm_result" in answer.columns
    series = answer["ml_generate_text_llm_result"]
    assert series[0].startswith(
        " BigFrames is a distributed computing framework"
        + " for processing massive data sets.")


def test_bigframeschain_chained_run() -> None:
    """Test valid call to bigframesllm."""
    session = bigframes_session()
    llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
    template = """Question: {question}
    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = BigFramesChain(prompt=prompt, llm=llm)

    # answer is a BigFrames DataFrame
    answer = llm_chain.run("What is BigFrames?")

    print(answer)
    assert "ml_generate_text_llm_result" in answer.columns
    series = answer["ml_generate_text_llm_result"]
    assert series[0].startswith(
        " BigFrames is a distributed computing framework"
        + " for processing massive data sets.")


def test_bigframesllmchained_df_input_chained_run() -> None:
    """Test valid call to bigframesllm."""
    session = bigframes_session()
    llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
    template = """Generate Pandas sample code for DataFrame.{api_name}"""

    prompt = PromptTemplate(template=template, input_variables=["api_name"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    ## read input from GCS
    df_api = bf.read_csv("gs://cloud-samples-data/vertex-ai/bigframe/df.csv")

    # answer is a BigFrames DataFrame
    answer = llm_chain(df_api["API"])
    # assert "ml_generate_text_llm_result" in answer['text'].columns
    series = answer['text']
    print(type(series))
    print(series)


# def test_bigframesllm_chained_batch() -> None:
#     """Test valid call to bigframesllm."""
#     session = bigframes_session()
#     llm = BigFramesLLM(session=session, connection=TEST_CONNECTION)
#     template = """Question: {question}
#     Answer: Let's think step by step."""

#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
    # answer is a bigframes dataframe
    # answer = llm_chain.batch([{"question":"What is BigFrames?"},
    #                           {"question":"What is BigQuery?"}])
    # print(answer)



    
    