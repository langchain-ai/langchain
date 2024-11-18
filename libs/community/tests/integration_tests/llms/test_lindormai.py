"""Test LindormAI API wrapper."""
from langchain_core.outputs import LLMResult

from langchain_community.llms.lindormai import LindormAILLM
import environs

env = environs.Env()
env.read_env(".env")


class Config:

    AI_LLM_ENDPOINT = env.str("AI_LLM_ENDPOINT", '<LLM_ENDPOINT>')
    AI_EMB_ENDPOINT = env.str("AI_EMB_ENDPOINT", '<EMB_ENDPOINT>')
    AI_USERNAME = env.str("AI_USERNAME", 'root')
    AI_PWD = env.str("AI_PWD", '<PASSWORD>')


    AI_CHAT_LLM_ENDPOINT = env.str("AI_CHAT_LLM_ENDPOINT", '<CHAT_ENDPOINT>')
    AI_CHAT_USERNAME = env.str("AI_CHAT_USERNAME", 'root')
    AI_CHAT_PWD = env.str("AI_CHAT_PWD", '<PASSWORD>')

    AI_DEFAULT_CHAT_MODEL = "qa_model_qwen_72b_chat"
    AI_DEFAULT_RERANK_MODEL = "rerank_bge_large"
    AI_DEFAULT_EMBEDDING_MODEL = "bge-large-zh-v1.5"
    AI_DEFAULT_XIAOBU2_EMBEDDING_MODEL = "xiaobu2"
    SEARCH_ENDPOINT = env.str("SEARCH_ENDPOINT", 'SEARCH_ENDPOINT')
    SEARCH_USERNAME = env.str("SEARCH_USERNAME", 'root')
    SEARCH_PWD = env.str("SEARCH_PWD", '<PASSWORD>')


def test_lindormai_call() -> None:
    """Test valid call to LindormAI."""
    llm = LindormAILLM(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD)  # type: ignore[call-arg]
    output = llm.invoke("who are you")

    print("invoke:", output)
    assert isinstance(output, str)


def test_lindormai_generate() -> None:
    """Test valid call to LindormAI."""
    llm = LindormAILLM(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD)  # type: ignore[call-arg]
    output = llm.generate(["who are you"])

    print("generate:", output)
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_lindormai_generate_stream() -> None:
    """Test valid call to LindormAI."""
    llm = LindormAILLM(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD,
        streaming=True
    )  # Provide necessary arguments
    output = llm.generate(["who are you"])

    print("stream:", output)  # noqa: T201
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


if __name__ == "__main__":
    test_lindormai_call()
    test_lindormai_generate()
    test_lindormai_generate_stream()
