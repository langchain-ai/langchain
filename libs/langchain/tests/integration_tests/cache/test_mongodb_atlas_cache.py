import os
import uuid

import pytest
from langchain.cache import MongoDBAtlasCache, MongoDBAtlasSemanticCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.messages import HumanMessage, AIMessage
from langchain.globals import get_llm_cache, set_llm_cache
from tests.integration_tests.cache.fake_embeddings import (
    ConsistentFakeEmbeddings,
)
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM

DEFAULT = "mongodb://localhost:62754/?directConnection=true"
CONN_STRING = os.environ.get("MONGODB_ATLAS_URI", DEFAULT)
COLLECTION = "default"
DATABASE = "default"


def random_string() -> str:
    return str(uuid.uuid4())


def llm_cache(cls):
    set_llm_cache(
        cls(
            embedding=ConsistentFakeEmbeddings(dimensionality=1536),
            connection_string=CONN_STRING,
            collection_name=COLLECTION,
            database_name=DATABASE,
            wait_until_ready=True,
        )
    )
    assert get_llm_cache()
    return get_llm_cache()


def _execute_test(prompt, llm, response):
    # Fabricate an LLM String
    llm_string = llm

    if not isinstance(llm, str):
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))

    # If the prompt is a str then we should pass just the string
    dumped_prompt = prompt if isinstance(prompt, str) else dumps(prompt)

    # Update the cache
    get_llm_cache().update(dumped_prompt, llm_string, response)

    # Retrieve the cached result through 'generate' call
    if isinstance(llm, str):
        output = get_llm_cache().lookup(prompt, llm)
        expected_output = response
    else:
        output = llm.generate([prompt])
        expected_output = LLMResult(
            generations=[response],
            llm_output={},
        )

    assert output == expected_output


@pytest.mark.parametrize(
    "prompt, llm, response",
    [
        ("foo", "bar", [Generation(text="fizz")]),
        ("foo", FakeLLM(), [Generation(text="fizz")]),
        (
            [HumanMessage(content="foo")],
            FakeChatModel(),
            [ChatGeneration(message=AIMessage(content="foo"))],
        ),
    ],
    ids=[
        "plain_cache",
        "cache_with_llm",
        "cache_with_chat",
    ],
)
@pytest.mark.parametrize("cacher", [MongoDBAtlasCache, MongoDBAtlasSemanticCache])
def test_mongodb_cache(cacher, prompt, llm, response):
    llm_cache(cacher)
    try:
        _execute_test(prompt, llm, response)
    finally:
        get_llm_cache().clear()


@pytest.mark.parametrize("embedding", [ConsistentFakeEmbeddings(dimensionality=1536)])
@pytest.mark.parametrize(
    "prompts, generations",
    [
        # Single prompt, single generation
        ([random_string()], [[random_string()]]),
        # Single prompt, multiple generations
        ([random_string()], [[random_string(), random_string()]]),
        # Single prompt, multiple generations
        ([random_string()], [[random_string(), random_string(), random_string()]]),
        # Multiple prompts, multiple generations
        (
            [random_string(), random_string()],
            [[random_string()], [random_string(), random_string()]],
        ),
    ],
    ids=[
        "single_prompt_single_generation",
        "single_prompt_two_generations",
        "single_prompt_three_generations",
        "multiple_prompts_multiple_generations",
    ],
)
def test_mongodb_atlas_cache_matrix(
    embedding: Embeddings,
    prompts: list[str],
    generations: list[list[str]],
):
    llm_cache(MongoDBAtlasSemanticCache)
    llm = FakeLLM()

    # Fabricate an LLM String
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))

    llm_generations = [
        [
            Generation(text=generation, generation_info=params)
            for generation in prompt_i_generations
        ]
        for prompt_i_generations in generations
    ]

    for prompt_i, llm_generations_i in zip(prompts, llm_generations):
        _execute_test(prompt_i, llm_string, llm_generations_i)
    assert llm.generate(prompts) == LLMResult(
        generations=llm_generations, llm_output={}
    )
    get_llm_cache().clear()
