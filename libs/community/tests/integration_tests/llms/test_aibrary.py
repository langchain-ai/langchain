"""Test AiBrary API wrapper.

In order to run this test, you need to have an AiBrary api key.
You can get it by registering for free at https://www.aibrary.dev/.
A test key can be found at https://www.aibrary.dev/dashboard/apikey

You'll then need to set AIBRARY_API_KEY environment variable to your api key.
"""

import pytest as pytest

from langchain_community.llms.aibrary import AiBrary
from typing import Generator


def test_aibrary_call() -> None:
    """Test simple call to AiBrary."""
    llm = AiBrary(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=250,
    )
    print(llm)
    output = llm.invoke("Say foo:")
    assert llm._llm_type == "aibrary"
    assert isinstance(output, str)

def test_aibrary_streaming() -> None:
    """Test streaming tokens from AiBrary."""
    llm = AiBrary(
        max_tokens=10,
        model="gpt-4o",
        temperature=0.2,
    )
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)