from unittest.mock import MagicMock

from langchain_community.tools.gmail.send_message import GmailSendMessage


def test_send() -> None:
    """Test gmail send."""
    mock_api_resource = MagicMock()
    # bypass pydantic validation as google-api-python-client is not a package dependency
    tool = GmailSendMessage.construct(api_resource=mock_api_resource)
    tool_input = {
        "to": "fake123@email.com",
        "subject": "subject line",
        "message": "message body",
    }
    result = tool.run(tool_input)
    assert result.startswith("Message sent. Message Id:")
    assert tool.args_schema is not None
