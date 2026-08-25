

def test_markdown_list_parser_plus_bullet() -> None:
    """Markdown `+` bullets must be parsed alongside `-` and `*`."""
    parser = MarkdownListOutputParser()
    result = parser.parse("+ foo\n+ bar\n- baz")
    assert result == ["foo", "bar", "baz"]


def test_markdown_list_parser_plus_bullet() -> None:
    """Markdown `+` bullets must be parsed alongside `-` and `*`."""
    parser = MarkdownListOutputParser()
    result = parser.parse("+ foo\n+ bar\n- baz")
    assert result == ["foo", "bar", "baz"]
