from pathlib import Path

import pytest

from langchain_prompty import create_chat_prompt

PROMPT_DIR = Path(__file__).parent / "prompts"


def test_double_templating() -> None:
    """
    Assess whether double templating occurs when invoking a chat prompt.
    If it does, an error is thrown and the test fails.
    """

    prompt_path = PROMPT_DIR / "double_templating.prompty"
    templated_prompt = create_chat_prompt(str(prompt_path))
    query = "What do you think of this JSON object: {'key': 7}?"

    try:
        templated_prompt.invoke(input={"user_input": query})
    except KeyError as e:
        pytest.fail("Double templating occurred: " + str(e))
