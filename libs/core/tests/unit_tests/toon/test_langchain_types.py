"""Tests for LangChain-specific type encoding."""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.toon import encode


def test_encode_human_message() -> None:
    """Test encoding a HumanMessage."""
    msg = HumanMessage(content="Hello, world!")
    result = encode(msg)

    assert "type: human" in result
    # Content with punctuation gets quoted
    assert 'content: "Hello, world!"' in result


def test_encode_ai_message() -> None:
    """Test encoding an AIMessage."""
    msg = AIMessage(content="I am an AI assistant.")
    result = encode(msg)

    assert "type: ai" in result
    # Content with spaces gets quoted
    assert 'content: "I am an AI assistant."' in result


def test_encode_system_message() -> None:
    """Test encoding a SystemMessage."""
    msg = SystemMessage(content="System prompt here")
    result = encode(msg)

    assert "type: system" in result
    # Content with spaces gets quoted
    assert 'content: "System prompt here"' in result


def test_encode_message_with_multiline_content() -> None:
    """Test encoding message with multiline content."""
    msg = HumanMessage(content="Line 1\nLine 2\nLine 3")
    result = encode(msg)

    assert "type: human" in result
    assert '"Line 1\\nLine 2\\nLine 3"' in result


def test_encode_message_with_name() -> None:
    """Test encoding message with name field."""
    msg = HumanMessage(content="Hello", name="Alice")
    result = encode(msg)

    assert "type: human" in result
    assert "content: Hello" in result
    assert "name: Alice" in result


def test_encode_message_with_additional_kwargs() -> None:
    """Test encoding message with additional_kwargs."""
    msg = AIMessage(content="Response", additional_kwargs={"model": "gpt-4"})
    result = encode(msg)

    assert "type: ai" in result
    assert "content: Response" in result
    assert "additional_kwargs:" in result
    assert "model: gpt-4" in result


def test_encode_array_of_messages() -> None:
    """Test encoding array of messages (uses tabular format)."""
    conversation = [
        HumanMessage(content="What is TOON?"),
        AIMessage(content="Token-Oriented Object Notation"),
    ]
    result = encode(conversation)
    lines = result.split("\n")

    # Messages with uniform primitive fields use tabular format
    assert lines[0] == "[2]{type,content}:"
    # Strings with spaces/punctuation get quoted
    assert lines[1] == '  human,"What is TOON?"'
    assert lines[2] == '  ai,"Token-Oriented Object Notation"'


def test_encode_document() -> None:
    """Test encoding a Document."""
    doc = Document(page_content="This is the content", metadata={"source": "test.txt"})
    result = encode(doc)

    # Content with spaces gets quoted
    assert 'page_content: "This is the content"' in result
    assert "metadata:" in result
    assert "source: test.txt" in result


def test_encode_document_without_metadata() -> None:
    """Test encoding document with empty metadata."""
    doc = Document(page_content="Simple content")
    result = encode(doc)

    # Content with spaces gets quoted
    assert 'page_content: "Simple content"' in result


def test_encode_document_with_multiline_content() -> None:
    """Test encoding document with multiline page_content."""
    doc = Document(
        page_content="Line 1\nLine 2\nLine 3", metadata={"source": "file.txt"}
    )
    result = encode(doc)

    assert "page_content:" in result
    assert '"Line 1\\nLine 2\\nLine 3"' in result


def test_encode_array_of_documents() -> None:
    """Test encoding array of documents."""
    docs = [
        Document(page_content="Doc 1", metadata={"id": 1}),
        Document(page_content="Doc 2", metadata={"id": 2}),
    ]
    result = encode(docs)

    assert "[2]:" in result
    # Content with spaces gets quoted
    assert 'page_content: "Doc 1"' in result
    assert 'page_content: "Doc 2"' in result


def test_encode_message_in_object() -> None:
    """Test encoding message as part of an object (tabular format)."""
    obj = {
        "conversation": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ],
        "metadata": {"user_id": 123},
    }
    result = encode(obj)

    # Messages use tabular format when they have uniform primitive fields
    assert "conversation[2]{type,content}:" in result
    assert "human,Hello" in result
    # "Hi there" has a space so it gets quoted
    assert 'ai,"Hi there"' in result
    assert "metadata:" in result
    assert "user_id: 123" in result


def test_encode_document_in_object() -> None:
    """Test encoding document as part of an object (tabular format)."""
    obj = {
        "documents": [
            Document(page_content="Content 1"),
            Document(page_content="Content 2"),
        ],
        "count": 2,
    }
    result = encode(obj)

    # Documents with only page_content use tabular format
    assert "documents[2]{page_content}:" in result
    assert "Content 1" in result
    assert "Content 2" in result
    assert "count: 2" in result


def test_encode_complex_message_structure() -> None:
    """Test encoding complex nested structure with messages (tabular format)."""
    data = {
        "session_id": "abc123",
        "messages": [
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
        ],
        "metadata": {"timestamp": "2024-01-01", "user": "alice"},
    }
    result = encode(data)

    assert "session_id: abc123" in result
    # Messages use tabular format
    assert "messages[3]{type,content}:" in result
    assert "Question 1" in result
    assert "Answer 1" in result
    assert "Question 2" in result
    assert "metadata:" in result
