"""Test Cassandra caches. Requires a running vector-capable Cassandra cluster."""
import os
import time
from typing import Any, Iterator, Tuple

import pytest

from langchain.cache import CassandraCache, CassandraSemanticCache
from langchain.globals import get_llm_cache, set_llm_cache
from langchain.schema import Generation, LLMResult
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.fixture(scope="module")
def cassandra_connection() -> Iterator[Tuple[Any, str]]:
    from cassandra.cluster import Cluster

    keyspace = "langchain_cache_test_keyspace"
    # get db connection
    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = os.environ["CONTACT_POINTS"].split(",")
        cluster = Cluster(contact_points)
    else:
        cluster = Cluster()
    #
    session = cluster.connect()
    # ensure keyspace exists
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )

    yield (session, keyspace)


def test_cassandra_cache(cassandra_connection: Tuple[Any, str]) -> None:
    session, keyspace = cassandra_connection
    cache = CassandraCache(session=session, keyspace=keyspace)
    set_llm_cache(cache)
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo"])
    print(output)
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    print(expected_output)
    assert output == expected_output
    cache.clear()


def test_cassandra_cache_ttl(cassandra_connection: Tuple[Any, str]) -> None:
    session, keyspace = cassandra_connection
    cache = CassandraCache(session=session, keyspace=keyspace, ttl_seconds=2)
    set_llm_cache(cache)
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    output = llm.generate(["foo"])
    assert output == expected_output
    time.sleep(2.5)
    # entry has expired away.
    output = llm.generate(["foo"])
    assert output != expected_output
    cache.clear()


def test_cassandra_semantic_cache(cassandra_connection: Tuple[Any, str]) -> None:
    session, keyspace = cassandra_connection
    sem_cache = CassandraSemanticCache(
        session=session,
        keyspace=keyspace,
        embedding=FakeEmbeddings(),
    )
    set_llm_cache(sem_cache)
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["bar"])  # same embedding as 'foo'
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    sem_cache.clear()
    output = llm.generate(["bar"])  # 'fizz' is erased away now
    assert output != expected_output
    sem_cache.clear()
