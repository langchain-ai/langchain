from pydantic import BaseModel, Field

from langchain_community.chat_models import ChatLlamaCpp


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# TODO: replace with standard integration tests
# See example in tests/integration_tests/chat_models/test_litellm.py
def test_structured_output() -> None:
    llm = ChatLlamaCpp(model_path="/path/to/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf")
    structured_llm = llm.with_structured_output(Joke)
    result = structured_llm.invoke("Tell me a short joke about cats.")
    assert isinstance(result, Joke)
