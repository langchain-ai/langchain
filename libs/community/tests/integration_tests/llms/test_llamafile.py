import os
from typing import Generator

import pytest
import requests
from requests.exceptions import ConnectionError, HTTPError

from langchain_community.llms.llamafile import Llamafile

LLAMAFILE_SERVER_BASE_URL = os.getenv(
    "LLAMAFILE_SERVER_BASE_URL", "http://localhost:8080"
)


def _ping_llamafile_server() -> bool:
    try:
        response = requests.get(LLAMAFILE_SERVER_BASE_URL)
        response.raise_for_status()
    except (ConnectionError, HTTPError):
        return False

    return True


@pytest.mark.skipif(
    not _ping_llamafile_server(),
    reason=f"unable to find llamafile server at {LLAMAFILE_SERVER_BASE_URL}, "
    f"please start one and re-run this test",
)
def test_llamafile_call() -> None:
    llm = Llamafile()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


@pytest.mark.skipif(
    not _ping_llamafile_server(),
    reason=f"unable to find llamafile server at {LLAMAFILE_SERVER_BASE_URL}, "
    f"please start one and re-run this test",
)
def test_llamafile_streaming() -> None:
    llm = Llamafile(streaming=True)
    generator = llm.stream("Tell me about Roman dodecahedrons.")
    assert isinstance(generator, Generator)
    for token in generator:
        assert isinstance(token, str)
