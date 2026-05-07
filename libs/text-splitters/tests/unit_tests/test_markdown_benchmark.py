"""Performance test for markdown splitters using pytest."""

from __future__ import annotations

import time

import pytest

from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter


def generate_markdown(num_sections: int) -> str:
    """Generate markdown document for testing."""
    content = "# Main Header\n\n"
    for i in range(num_sections):
        content += f"## Section {i + 1}\n\nContent for section {i + 1}.\n\n"
        for j in range(3):
            content += f"### Subsection {i + 1}.{j + 1}\n\nDetails here.\n\n"
    return content


@pytest.mark.benchmark
def test_experimental_markdown_splitter_performance_small() -> None:
    """Test performance with small document (10 sections)."""
    splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )
    markdown_text = generate_markdown(10)

    start = time.perf_counter()
    chunks = splitter.split_text(markdown_text)
    elapsed = time.perf_counter() - start

    assert len(chunks) > 0
    assert elapsed < 0.01  # Should complete in less than 10ms


@pytest.mark.benchmark
def test_experimental_markdown_splitter_performance_medium() -> None:
    """Test performance with medium document (100 sections)."""
    splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )
    markdown_text = generate_markdown(100)

    start = time.perf_counter()
    chunks = splitter.split_text(markdown_text)
    elapsed = time.perf_counter() - start

    assert len(chunks) > 0
    assert elapsed < 0.05  # Should complete in less than 50ms


@pytest.mark.benchmark
def test_experimental_markdown_splitter_performance_large() -> None:
    """Test performance with large document (500 sections)."""
    splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
    )
    markdown_text = generate_markdown(500)

    start = time.perf_counter()
    chunks = splitter.split_text(markdown_text)
    elapsed = time.perf_counter() - start

    assert len(chunks) > 0
    # Linear time complexity - should scale proportionally
    assert elapsed < 0.3  # Should complete in less than 300ms
