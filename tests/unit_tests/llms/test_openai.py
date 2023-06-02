import uuid

import pytest

from langchain import OpenAI


@pytest.mark.requires("openai")
def test_module_var_not_set():
    import openai

    key = str(uuid.uuid4())
    OpenAI(openai_api_key=key, pass_creds_at_invocation=True)
    assert openai.api_key != key
