"""Test WatsonxEmbeddings.

You'll need to set WATSONX_APIKEY and WATSONX_PROJECT_ID environment variables.
"""

import os

from ibm_watsonx_ai import APIClient  # type: ignore
from langchain_community.vectorstores.chroma import Chroma

from langchain_ibm import WatsonxEmbeddings

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "ibm/slate-125m-english-rtrvr"

DOCUMENTS = ["What is a generative ai?", "What is a loan and how does it works?"]


def test_01_generate_embed_documents() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID
    )
    generate_embedding = watsonx_embedding.embed_documents(texts=DOCUMENTS)
    assert len(generate_embedding) == len(DOCUMENTS)
    assert all(isinstance(el, float) for el in generate_embedding[0])


def test_02_generate_embed_query() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    generate_embedding = watsonx_embedding.embed_query(text=DOCUMENTS[0])
    assert isinstance(generate_embedding, list) and isinstance(
        generate_embedding[0], float
    )


def test_10_pass_client() -> None:
    watsonx_client = APIClient(
        wml_credentials={
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


def test_90_generate_embed_chroma_integration() -> None:
    watsonx_embedding = WatsonxEmbeddings(
        model_id=MODEL_ID,
        url=URL,
        project_id=WX_PROJECT_ID,
    )
    vectorstore = Chroma.from_texts(
        texts=[
            "harrison worked at kensho",
            "I have blue eye's",
            "My name is Mateusz",
            "I got 5 at math in school",
            "My best friend is Lukas",
        ],
        collection_name="rag-chroma",
        embedding=watsonx_embedding,
    )
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query="What is my best grade in school?")

    assert docs
    assert isinstance(docs, list)
    assert getattr(docs[0], "page_content", None)
