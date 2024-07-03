import logging
import logging.config
import os

import pytest
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")


def test_openai_demo_1_completions(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore
    client = OpenAI(
        # api_key="YOUR_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[{"role": "user", "content": "你好"}],
        top_p=0.7,
        temperature=0.1,
        max_tokens=2000,
    )
    logger.info("\033[1;32m" + f"client: {response}" + "\033[0m")


def test_openai_demo_1_embeddings(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore
    client = OpenAI(
        # api_key="YOUR_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )
    response = client.embeddings.create(
        model="embedding-2",
        input="你好",
    )
    logger.info("\033[1;32m" + f"client: {response}" + "\033[0m")


@pytest.mark.scheduled
def test_openai_model_glm4(logging_conf) -> None:
    """Test zhipuai model"""
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm = ChatOpenAI(
        model_name="glm-4-0520",
        # openai_api_key="YOUR_API_KEY",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    )
    template = """Question: {question}
    
    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    responses = llm_chain.run("你好")
    logger.info("\033[1;32m" + f"llm_chain: {responses}" + "\033[0m")


@pytest.mark.scheduled
def test_openai_embedding_documents(logging_conf) -> None:
    """Test zhipuai embeddings."""
    logging.config.dictConfig(logging_conf)  # type: ignore
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings(
        model="embedding-2", openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    # TODO: _aget_len_safe_embeddings会使用cl100k_base的模型编码成tokener,语义无法对齐，需要重写这块逻辑
    output = embedding.embed_documents(documents)
    logger.info("\033[1;32m" + f"embedding: {output}" + "\033[0m")
    assert len(output) == 1
    assert len(output[0]) == 1024
