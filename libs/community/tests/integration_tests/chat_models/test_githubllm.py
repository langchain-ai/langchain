import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models.githubllm import GithubLLM  # Replace 'your_module' with the actual module name
import requests

# Set up environment variables for testing
os.environ["GITHUB_TOKEN"] = "fake_github_token"
os.environ["AZURE_API_KEY"] = "fake_azure_key"

@pytest.fixture
def github_llm():
    return GithubLLM(model="gpt-4o", use_azure_fallback=True)

def test_github_llm_initialization(github_llm):
    assert github_llm.model == "gpt-4o"
    assert github_llm.use_azure_fallback == True
    assert github_llm.github_api_key.get_secret_value() == "fake_github_token"
    assert github_llm.azure_api_key.get_secret_value() == "fake_azure_key"

def test_convert_message_to_dict():
    from langchain_community.chat_models.githubllm import _convert_message_to_dict  # Import the function

    system_message = SystemMessage(content="You are a helpful assistant.")
    human_message = HumanMessage(content="Hello, how are you?")
    ai_message = AIMessage(content="I'm doing well, thank you!")

    assert _convert_message_to_dict(system_message) == {"role": "system", "content": "You are a helpful assistant."}
    assert _convert_message_to_dict(human_message) == {"role": "user", "content": "Hello, how are you?"}
    assert _convert_message_to_dict(ai_message) == {"role": "assistant", "content": "I'm doing well, thank you!"}

def test_convert_dict_to_message():
    from langchain_community.chat_models.githubllm import _convert_dict_to_message  # Import the function

    system_dict = {"role": "system", "content": "You are a helpful assistant."}
    human_dict = {"role": "user", "content": "Hello, how are you?"}
    ai_dict = {"role": "assistant", "content": "I'm doing well, thank you!"}

    assert isinstance(_convert_dict_to_message(system_dict), SystemMessage)
    assert isinstance(_convert_dict_to_message(human_dict), HumanMessage)
    assert isinstance(_convert_dict_to_message(ai_dict), AIMessage)

@patch('requests.post')
def test_github_llm_generate(mock_post, github_llm):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is a test response."}}]
    }
    mock_post.return_value = mock_response

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Tell me a joke.")
    ]

    result = github_llm._generate(messages)

    assert len(result.generations) == 1
    assert result.generations[0].message.content == "This is a test response."

@patch('requests.post')
def test_github_llm_stream(mock_post, github_llm):
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        b'{"choices": [{"delta": {"content": "This"}}]}',
        b'{"choices": [{"delta": {"content": " is"}}]}',
        b'{"choices": [{"delta": {"content": " a"}}]}',
        b'{"choices": [{"delta": {"content": " test."}}]}',
    ]
    mock_post.return_value = mock_response

    messages = [HumanMessage(content="Tell me a joke.")]

    chunks = list(github_llm._stream(messages))

    assert len(chunks) == 4
    assert "".join(chunk.message.content for chunk in chunks) == "This is a test."

@patch('requests.post')
def test_github_llm_azure_fallback(mock_post, github_llm):
    # Simulate GitHub API failure
    mock_post.side_effect = [
        requests.exceptions.RequestException("GitHub API error"),
        MagicMock(json=lambda: {"choices": [{"message": {"content": "Azure fallback response."}}]})
    ]

    messages = [HumanMessage(content="Tell me a joke.")]

    result = github_llm._generate(messages)

    assert len(result.generations) == 1
    assert result.generations[0].message.content == "Azure fallback response."

def test_github_llm_rate_limit():
    llm = GithubLLM(model="gpt-4o", max_requests_per_minute=1)

    assert llm._check_rate_limit() == True
    llm._increment_request_count()
    assert llm._check_rate_limit() == False

if __name__ == "__main__":
    pytest.main()