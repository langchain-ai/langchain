from langchain import utils


def test_truncate_word():
    assert utils.truncate_word("Hello World", 5) == "He..."
    assert utils.truncate_word("Hello World", 5, "!!!") == "He!!!"
    assert utils.truncate_word("Hello World", 12, "!!!") == "Hello World"
