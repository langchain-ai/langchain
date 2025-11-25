from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from langchain_ollama.chat_models import ChatOllama


def test_reasoning_param_passed_to_client() -> None:
    """Test that the reasoning parameter is correctly passed to the Ollama client."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "message": {"role": "assistant", "content": "I am thinking..."},
                "done": True,
                "done_reason": "stop",
            }
        ]

        # Case 1: reasoning=True in init
        llm = ChatOllama(model="deepseek-r1", reasoning=True)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is True

        # Case 2: reasoning=False in init
        llm = ChatOllama(model="deepseek-r1", reasoning=False)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is False

        # Case 3: reasoning passed in invoke
        llm = ChatOllama(model="deepseek-r1")
        llm.invoke([HumanMessage("Hello")], reasoning=True)

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is True
