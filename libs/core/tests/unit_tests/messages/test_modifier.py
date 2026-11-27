import pytest

from langchain_core.messages import RemoveMessage


@pytest.mark.parametrize("content", ["", None])
def test_remove_message_rejects_falsy_content(content: str | None) -> None:
    with pytest.raises(ValueError, match="content"):
        RemoveMessage(id="message_to_remove", content=content)
