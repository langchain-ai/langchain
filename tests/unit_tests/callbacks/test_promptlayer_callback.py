from unittest.mock import MagicMock
from langchain.llms import OpenAI, Replicate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.promptlayer_callback import PromptLayerHandler
from langchain.schema import HumanMessage, LLMResult

# Mocking external classes / functions
request_id_func = MagicMock()
HumanMessage = MagicMock()


@pytest.fixture
def openai_llm() -> OpenAI:
    return OpenAI(
        model_name="text-davinci-002",
        callbacks=[PromptLayerHandler(
            request_id_func=request_id_func,
            pl_tags=["OPENAI WORKS!"]
        )]
    )


@pytest.fixture
def replicate_llm() -> Replicate:
    return Replicate(
        model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
        callbacks=[PromptLayerHandler(
            request_id_func=request_id_func,
            pl_tags=["REPLICATE WORKS!"]
        )]
    )


@pytest.fixture
def chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=0,
        callbacks=[PromptLayerHandler(
            request_id_func=request_id_func,
            pl_tags=["OpenAIChat WORKS!"]
        )]
    )


def test_openai_llm_generate(openai_llm: OpenAI) -> None:
    responses = openai_llm.generate([
        "Tell me a joke 1",
        "Where is Ohio? 2",
        "Where is Ohio? 3",
    ])
    assert isinstance(responses, list), "Response should be a list"
    assert len(responses) == 3, "Should return three responses"
    assert all(isinstance(response, LLMResult) for response in responses), "All responses should be of type LLMResult"


def test_replicate_llm_generate(replicate_llm: Replicate) -> None:
    responses = replicate_llm.generate([
        "Tell me a joke 1",
        "Where is Ohio? 2",
        "Where is Ohio? 3",
    ])
    assert isinstance(responses, list), "Response should be a list"
    assert len(responses) == 3, "Should return three responses"
    assert all(isinstance(response, LLMResult) for response in responses), "All responses should be of type LLMResult"


def test_chat_llm_generate(chat_llm: ChatOpenAI) -> None:
    responses = chat_llm([
        HumanMessage(content="What comes after 1,2,3 ?"),
        HumanMessage(content="Tell me another joke?"),
    ])
    assert isinstance(responses, list), "Response should be a list"
    assert len(responses) == 2, "Should return two responses"
    assert all(isinstance(response, LLMResult) for response in responses), "All responses should be of type LLMResult"
