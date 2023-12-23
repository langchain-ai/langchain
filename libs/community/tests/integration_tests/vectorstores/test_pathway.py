"""Test Pathway vector store functionality."""

import pathlib
import sys
import time
from multiprocessing import Process

import pytest
import requests

from langchain_community.vectorstores.pathway import (
    PathwayVectorClient,
    PathwayVectorServer,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8764


def pathway_server(tmp_path):
    import pathway as pw

    data_sources = []
    data_sources.append(
        pw.io.fs.read(
            tmp_path,
            format="binary",
            mode="streaming",
            with_metadata=True,
        )
    )

    embeddings_model = FakeEmbeddings()

    vector_server = PathwayVectorServer(*data_sources, embedder=embeddings_model)
    thread = vector_server.run_server(
        host=PATHWAY_HOST,
        port=PATHWAY_PORT,
        threaded=True,
        with_cache=False,
    )
    thread.join()


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Pathway requires python 3.10 or higher",
)
@pytest.mark.requires("pathway")
def test_similarity_search_without_metadata(tmp_path: pathlib.Path) -> None:
    with open(tmp_path / "file_one.txt", "w+") as f:
        f.write("foo")

    p = Process(target=pathway_server, args=[tmp_path])
    p.start()
    time.sleep(5)
    client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)
    MAX_ATTEMPTS = 5
    attempts = 0
    output = []
    while attempts < MAX_ATTEMPTS:
        try:
            output = client.similarity_search("foo")
        except requests.exceptions.RequestException:
            pass
        else:
            break
        time.sleep(1)
        attempts += 1
    p.terminate()
    time.sleep(2)
    assert len(output) == 1
    assert output[0].page_content == "foo"
