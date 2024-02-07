from typing import Generator

import pytest

from langchain_community.llms.llamafile import Llamafile


def test_llamafile_call():
    llm = Llamafile()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_llamafile_streaming():
    llm = Llamafile(streaming=True)
    generator = llm.stream("Tell me about Roman dodecahedrons.")
    assert isinstance(generator, Generator)
    for token in generator:
        assert isinstance(token, str)
