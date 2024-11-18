"""Test Lindorm AI Chat Model."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.lindormai import ChatLindormAI
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


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatLindormAI(model_name=Config.AI_DEFAULT_CHAT_MODEL,
                      endpoint=Config.AI_CHAT_LLM_ENDPOINT,
                      username=Config.AI_CHAT_USERNAME,
                      password=Config.AI_CHAT_PWD),
    ]:
        assert model.model_name == Config.AI_DEFAULT_CHAT_MODEL


def test_default_call() -> None:
    """Test default model call."""
    chat = ChatLindormAI(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD)  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    print("response: ", response)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatLindormAI(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD)  # type: ignore[call-arg]
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    print("response: ", response)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = ChatLindormAI(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD)  # type: ignore[call-arg]
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        print("generations: ", generations)
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


if __name__ == "__main__":
    test_initialization()
    test_default_call()
    test_multiple_history()
    test_multiple_messages()
