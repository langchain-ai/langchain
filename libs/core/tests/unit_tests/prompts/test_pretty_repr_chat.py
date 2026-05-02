from langchain_core.prompts.chat import ChatPromptTemplate


def test_pretty_repr_text_includes_title_messages_and_partials() -> None:
    tmpl = ChatPromptTemplate(
        [
            ("system", "You are {name}."),
            ("human", "Hello"),
        ],
        partial_variables={"name": "Bob"},
    )

    s = tmpl.pretty_repr(html=False)

    assert "Chat Prompt" in s
    assert "Partials: name" in s
    # Message sections rendered by underlying prompt templates
    assert "System Message" in s
    assert "Human Message" in s


def test_pretty_repr_html_includes_title_and_partials() -> None:
    tmpl = ChatPromptTemplate(
        [
            ("system", "You are {name}."),
            ("human", "Hello"),
        ],
        partial_variables={"name": "Bob"},
    )

    s = tmpl.pretty_repr(html=True)

    assert "Chat Prompt" in s
    assert "Partials:" in s
    # Do not rely on exact color/HTML codes; ensure key presence
    assert "name" in s
