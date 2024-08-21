import os
import uuid
from typing import Any, List, Union

import pytest  # type: ignore[import-not-found]
from langchain_core.caches import BaseCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.load.dump import dumps
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from langchain_mongodb.cache import MongoDBAtlasSemanticCache, MongoDBCache

from ..utils import ConsistentFakeEmbeddings, FakeChatModel, FakeLLM

CONN_STRING = os.environ.get("MONGODB_ATLAS_URI")
INDEX_NAME = "langchain-test-index-semantic-cache"
DATABASE = "langchain_test_db"
COLLECTION = "langchain_test_cache"


def random_string() -> str:
    return str(uuid.uuid4())


def llm_cache(cls: Any) -> BaseCache:
    set_llm_cache(
        cls(
            embedding=ConsistentFakeEmbeddings(dimensionality=1536),
            connection_string=CONN_STRING,
            collection_name=COLLECTION,
            database_name=DATABASE,
            index_name=INDEX_NAME,
            score_threshold=0.5,
            wait_until_ready=15.0,
        )
    )
    assert get_llm_cache()
    return get_llm_cache()


def _execute_test(
    prompt: Union[str, List[BaseMessage]],
    llm: Union[str, FakeLLM, FakeChatModel],
    response: List[Generation],
) -> None:
    # Fabricate an LLM String

    if not isinstance(llm, str):
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
    else:
        llm_string = llm

    # If the prompt is a str then we should pass just the string
    dumped_prompt: str = prompt if isinstance(prompt, str) else dumps(prompt)

    # Update the cache
    get_llm_cache().update(dumped_prompt, llm_string, response)

    # Retrieve the cached result through 'generate' call
    output: Union[List[Generation], LLMResult, None]
    expected_output: Union[List[Generation], LLMResult]

    if isinstance(llm, str):
        output = get_llm_cache().lookup(dumped_prompt, llm)  # type: ignore
        expected_output = response
    else:
        output = llm.generate([prompt])  # type: ignore
        expected_output = LLMResult(
            generations=[response],
            llm_output={},
        )

    assert output == expected_output  # type: ignore


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
@pytest.mark.parametrize("cacher", [MongoDBCache, MongoDBAtlasSemanticCache])
@pytest.mark.parametrize("remove_score", [True, False])
def test_mongodb_cache(
    remove_score: bool,
    cacher: Union[MongoDBCache, MongoDBAtlasSemanticCache],
    prompt: Union[str, List[BaseMessage]],
    llm: Union[str, FakeLLM, FakeChatModel],
    response: List[Generation],
) -> None:
    llm_cache(cacher)
    if remove_score:
        get_llm_cache().score_threshold = None  # type: ignore
    try:
        _execute_test(prompt, llm, response)
    finally:
        get_llm_cache().clear()


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
    prompts: List[str],
    generations: List[List[str]],
) -> None:
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
