"""Test WatsonxEmbeddings.

You'll need to set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.
"""

import os

from ibm_watsonx_ai import APIClient  # type: ignore
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames  # type: ignore

from langchain_ibm import WatsonxEmbeddings

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "ibm/slate-125m-english-rtrvr"

DOCUMENTS = ["What is a generative ai?", "What is a loan and how does it works?"]


def test_01_generate_embed_documents() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,  # type: ignore[arg-type]
    )
    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_02_generate_embed_query() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = watsonx_embedding.embed_query(text=DOCUMENTS[0])
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )


def test_03_generate_embed_documents_with_param() -> None:
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,  # type: ignore[arg-type]
        project_id=WX_PROJECT_ID,
        params=embed_params,  # type: ignore[arg-type]
    )
    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_10_generate_embed_query_with_client_initialization() -> None:
    watsonx_client = APIClient(
        credentials={
            "url": URL,
            "apikey": WX_APIKEY,
        }
    )
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID, project_id=WX_PROJECT_ID, watsonx_client=watsonx_client
    )
    generate_embedding = watsonx_embedding.embed_query(text=DOCUMENTS[0])
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )
