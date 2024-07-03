import logging
import logging.config

import numpy as np
import pytest
import zhipuai
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from zhipuai import ZhipuAI

from langchain_glm import ChatZhipuAI
from langchain_glm.embeddings.base import ZhipuAIEmbeddings

logger = logging.getLogger(__name__)


def test_demo_1_completions(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore
    client = ZhipuAI()  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-0520",
        messages=[{"role": "user", "content": "你好"}],
        top_p=0.7,
        temperature=0.1,
        max_tokens=2000,
    )
    logger.info("\033[1;32m" + f"client: {response}" + "\033[0m")


def test_demo_1_embeddings(logging_conf):
    logging.config.dictConfig(logging_conf)  # type: ignore
    client = ZhipuAI()  # 填写您自己的APIKey
    response = client.embeddings.create(
        model="embedding-2",
        input="你好",
    )
    logger.info("\033[1;32m" + f"client: {response}" + "\033[0m")


@pytest.mark.scheduled
def test_zhipuai_model_glm4(logging_conf) -> None:
    """Test zhipuai model"""
    logging.config.dictConfig(logging_conf)  # type: ignore

    llm = ChatZhipuAI(
        model_name="glm-4-0520",
        # openai_api_key="YOUR_API_KEY",
    )
    template = """Question: {question}
    
    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    responses = llm_chain.run("你好")
    logger.info("\033[1;32m" + f"llm_chain: {responses}" + "\033[0m")


@pytest.mark.scheduled
def test_zhipuai_embedding_documents(logging_conf) -> None:
    """Test zhipuai embeddings."""
    logging.config.dictConfig(logging_conf)  # type: ignore
    documents = ["foo bar"]
    embedding = ZhipuAIEmbeddings()
    output = embedding.embed_documents(documents)
    logger.info("\033[1;32m" + f"embedding: {output}" + "\033[0m")
    assert len(output) == 1
    assert len(output[0]) == 1024
