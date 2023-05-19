"""Test caching for LLMs and ChatModels."""
from typing import Dict, Union
import langchain
from langchain.cache import InMemoryCache
from langchain.chat_models import FakeListChatModel
from langchain.chat_models.base import chat_history_as_string, BaseChatModel
from langchain.llms import FakeListLLM
from langchain.llms.base import BaseLLM
from langchain.schema import Generation, ChatGeneration, _message_from_dict


def create_llm_string(llm: Union[BaseLLM, BaseChatModel]) -> str:
    _dict: Dict = llm.dict()
    _dict["stop"] = None
    return str(sorted([(k, v) for k, v in _dict.items()]))

def test_llm_caching() -> None:
    langchain.llm_cache = InMemoryCache()
    prompt = "How are you?"
    response = "Test response"
    cached_response = "Cached test response"
    llm = FakeListLLM(responses=[response])
    langchain.llm_cache.update(prompt=prompt,
                               llm_string=create_llm_string(llm),
                               return_val=[Generation(text=cached_response)])
    assert llm(prompt) == cached_response

def test_chat_model_caching() -> None:
    langchain.llm_cache = InMemoryCache()
    prompt = [_message_from_dict({
        "type": "human",
        "data": {"content": "How are you?"}
    })]
    response = "Test response"
    cached_response = "Cached test response"
    cached_message = _message_from_dict({
        "type": "ai",
        "data": {"content": cached_response}
    })
    llm = FakeListChatModel(responses=[response])
    langchain.llm_cache.update(prompt=chat_history_as_string(prompt),
                               llm_string=create_llm_string(llm),
                               return_val=[ChatGeneration(text=cached_response,
                                                          message=cached_message)])
    assert llm(prompt) == cached_response
