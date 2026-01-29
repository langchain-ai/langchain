"""Tests for ExperimentalMarkdownSyntaxTextSplitter unclosed code block handling.

Per the CommonMark specification, unclosed code blocks should be closed at the
end of the document, preserving the content.

Reference: https://spec.commonmark.org/0.31.2/#fenced-code-blocks
"If the end of the containing block (or document) is reached and no closing
code fence has been found, the code block contains all of the lines after
the opening code fence..."
"""

from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter


class TestExperimentalMarkdownUnclosedCodeBlock:
    """Test that unclosed code blocks preserve content per CommonMark spec."""

    def test_unclosed_code_block_preserves_content(self) -> None:
        """Unclosed code block content should be preserved per CommonMark spec."""
        splitter = ExperimentalMarkdownSyntaxTextSplitter()

        md = "text\n```\ncode"
        result = splitter.split_text(md)

        all_content = "".join(doc.page_content for doc in result)
        assert "code" in all_content

    def test_unclosed_code_block_preserves_all_content(self) -> None:
        """All content after unclosed code fence should be preserved."""
        splitter = ExperimentalMarkdownSyntaxTextSplitter()

        md = "start\n```\ncode1\nmore code\neven more"
        result = splitter.split_text(md)

        all_content = "".join(doc.page_content for doc in result)
        assert "code1" in all_content
        assert "more code" in all_content
        assert "even more" in all_content

    def test_closed_code_block_preserves_content(self) -> None:
        """Properly closed code blocks should preserve content."""
        splitter = ExperimentalMarkdownSyntaxTextSplitter()

        md = "text\n```\ncode\n```"
        result = splitter.split_text(md)

        all_content = "".join(doc.page_content for doc in result)
        assert "code" in all_content

    def test_mixed_closed_and_unclosed_code_blocks(self) -> None:
        """Both closed and unclosed code blocks should preserve content."""
        splitter = ExperimentalMarkdownSyntaxTextSplitter()

        md = "a\n```\nb\n```\nc\n```\nd"
        result = splitter.split_text(md)

        all_content = "".join(doc.page_content for doc in result)
        # First block (closed) preserved
        assert "b" in all_content
        # Second block (unclosed) also preserved
        assert "d" in all_content
