from langchain_openai.chat_models.base import _resize


def test_resize_smaller_side_scaled_correctly() -> None:
    width, height = _resize(1000, 2000)
    assert width == 768 and height == 1536
