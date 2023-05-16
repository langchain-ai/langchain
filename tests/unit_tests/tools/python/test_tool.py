from langchain.tools.python.tool import sanitize_input


def test_sanitize_input() -> None:
    query = """
    ```
        x = 5
    ```
    """
    expected = "x = 5"
    actual = sanitize_input(query)
    assert expected == actual

    query = """
       ```python
        x = 5
    ```
    """
    expected = "x = 5"
    actual = sanitize_input(query)
    assert expected == actual

    query = """
    x = 5
    """
    expected = "x = 5"
    actual = sanitize_input(query)
    assert expected == actual
