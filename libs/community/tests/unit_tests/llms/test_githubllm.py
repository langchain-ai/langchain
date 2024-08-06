import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_community.llms.github import GithubLLM

def test_githubllm_initialization():
    llm = GithubLLM(model="gpt-4o", system_prompt="You are a knowledgeable history teacher.")
    assert llm.model == "gpt-4o"
    assert llm.system_prompt == "You are a knowledgeable history teacher."

def test_githubllm_invalid_model():
    with pytest.raises(ValueError):
        GithubLLM(model="invalid-model")

@patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"})
@patch("requests.post")
def test_githubllm_call(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Paris is the capital of France."}}]
    }
    mock_post.return_value = mock_response

    llm = GithubLLM(model="gpt-4o")
    response = llm("What is the capital of France?")
    
    assert response == "Paris is the capital of France."
    mock_post.assert_called_once()

@patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"})
@patch("requests.post")
def test_githubllm_chat(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "The main causes of the French Revolution were..."}}]
    }
    mock_post.return_value = mock_response

    llm = GithubLLM(model="gpt-4o")
    conversation = [
        {"role": "user", "content": "Tell me about the French Revolution."},
        {"role": "assistant", "content": "The French Revolution was a period of major social and political upheaval in France..."},
        {"role": "user", "content": "What were the main causes?"}
    ]
    
    response = llm.chat(conversation)
    
    assert response == "The main causes of the French Revolution were..."
    mock_post.assert_called_once()

@patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token"})
@patch("requests.post")
def test_githubllm_stream(mock_post):
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [b"Chunk 1", b"Chunk 2", b"Chunk 3"]
    mock_post.return_value = mock_response

    llm = GithubLLM(model="gpt-4o")
    chunks = list(llm.stream("Can you elaborate on the Reign of Terror?"))
    
    assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]
    mock_post.assert_called_once()

def test_githubllm_missing_token():
    llm = GithubLLM(model="gpt-4o")
    with pytest.raises(ValueError, match="GITHUB_TOKEN environment variable is not set."):
        llm("This should fail without a token.")