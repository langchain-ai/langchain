from pathlib import Path

from langchain.document_loaders import FacebookChatLoader


def test_facebook_chat_loader() -> None:
    """Test FacebookChatLoader."""
    file_path = Path(__file__).parent.parent / "examples/facebook_chat.json"
    loader = FacebookChatLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(file_path)
    assert docs[0].page_content == (
        "User 2 on 2023-02-05 13:46:11: Bye!\n\n"
        "User 1 on 2023-02-05 13:43:55: Oh no worries! Bye\n\n"
        "User 2 on 2023-02-05 13:24:37: No Im sorry it was my mistake, "
        "the blue one is not for sale\n\n"
        "User 1 on 2023-02-05 13:05:40: I thought you were selling the blue one!\n\n"
        "User 1 on 2023-02-05 13:05:09: Im not interested in this bag. "
        "Im interested in the blue one!\n\n"
        "User 2 on 2023-02-05 13:04:28: Here is $129\n\n"
        "User 2 on 2023-02-05 13:04:05: Online is at least $100\n\n"
        "User 1 on 2023-02-05 12:59:59: How much do you want?\n\n"
        "User 2 on 2023-02-05 08:17:56: Goodmorning! $50 is too low.\n\n"
        "User 1 on 2023-02-05 00:17:02: Hi! Im interested in your bag. "
        "Im offering $50. Let me know if you are interested. Thanks!\n\n"
    )
