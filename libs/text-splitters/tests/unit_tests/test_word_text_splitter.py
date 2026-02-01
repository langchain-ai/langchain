from langchain_text_splitters.word import WordTextSplitter


def test_word_text_splitter_basic() -> None:
    """Test basic word splitting functionality."""
    text = "one two three four five"
    splitter = WordTextSplitter(chunk_size=3, chunk_overlap=0)
    chunks = splitter.split_text(text)
    assert chunks == ["one two three", "four five"]


def test_word_text_splitter_with_overlap() -> None:
    """Test word splitting with overlap."""
    text = "one two three four five"
    splitter = WordTextSplitter(chunk_size=3, chunk_overlap=1)
    chunks = splitter.split_text(text)
    assert chunks == ["one two three", "three four five"]


def test_word_text_splitter_custom_separator() -> None:
    """Test word splitting with custom separator."""
    text = "one-two-three-four-five"
    # Testing splitting with a custom separator isn't exactly what
    # WordTextSplitter does. WordTextSplitter splits by word pattern then joins
    # with separator.
    # So we need to ensure the splitting naturally finds "words" or we provide a
    # pattern.

    # If the text has no spaces, the default whitespace splitter will see one big word.
    splitter = WordTextSplitter(chunk_size=2, chunk_overlap=0)
    chunks = splitter.split_text(text)
    assert chunks == ["one-two-three-four-five"]

    # Use a pattern that splits non-hyphens
    splitter = WordTextSplitter(
        chunk_size=2, chunk_overlap=0, word_pattern=r"[^-\s]+", separator="-"
    )
    chunks = splitter.split_text(text)
    assert chunks == ["one-two", "three-four", "five"]


def test_word_text_splitter_custom_join_separator() -> None:
    """Test that we can split by space but join with something else."""
    text = "one two three four"
    splitter = WordTextSplitter(chunk_size=2, chunk_overlap=0, separator="-")
    chunks = splitter.split_text(text)
    assert chunks == ["one-two", "three-four"]


def test_word_text_splitter_empty() -> None:
    """Test validation with empty string."""
    splitter = WordTextSplitter(chunk_size=2, chunk_overlap=0)
    assert splitter.split_text("") == []


def test_word_text_splitter_small_text() -> None:
    """Test validation with text smaller than chunk size."""
    text = "one two"
    splitter = WordTextSplitter(chunk_size=5, chunk_overlap=0)
    assert splitter.split_text(text) == ["one two"]


def test_word_text_splitter_whitespace_handling() -> None:
    """Test that multiple whitespaces are handled correctly by default."""
    text = "one   two\tthree\nfour"
    splitter = WordTextSplitter(chunk_size=2, chunk_overlap=0)
    chunks = splitter.split_text(text)
    assert chunks == ["one two", "three four"]
