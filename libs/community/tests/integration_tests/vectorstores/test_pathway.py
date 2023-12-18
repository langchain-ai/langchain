"""Test Pathway vector store functionality."""
import os
import shutil
import sys
import time
from multiprocessing import Process

import pytest

from langchain_community.vectorstores.pathway import (
    PathwayVectorClient,
    PathwayVectorServer,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8754
PATHWAY_DIR = "/tmp/pathway-samples/"


def pathway_server():
    import pathway as pw

    data_sources = []
    data_sources.append(
        pw.io.fs.read(
            PATHWAY_DIR,
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
class TestPathway:
    @classmethod
    def setup_class(cls) -> None:
        os.mkdir(PATHWAY_DIR)
        with open(f"{PATHWAY_DIR}file_one.txt", "w+") as f:
            f.write("foo")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(PATHWAY_DIR)
        pass

    def test_similarity_search_without_metadata(self) -> None:
        p = Process(target=pathway_server)
        p.start()
        time.sleep(6)
        client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)
        output = client.similarity_search("foo")
        p.terminate()
        time.sleep(2)
        assert len(output) == 1
        assert output[0].page_content == "foo"
