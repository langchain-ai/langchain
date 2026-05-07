"""Performance benchmark for Markdown text splitters."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import langchain_text_splitters
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter

# Generate a large markdown document for testing
def generate_large_markdown(num_sections: int = 100) -> str:
    """Generate a large markdown document with multiple sections."""
    content = "# Main Header\n\n"
    content += "This is the introduction to the document.\n\n"

    for i in range(num_sections):
        content += f"## Section {i + 1}\n\n"
        content += f"Content for section {i + 1}.\n"
        content += f"This is paragraph 1 of section {i + 1}.\n"
        content += f"This is paragraph 2 of section {i + 1}.\n\n"

        for j in range(3):
            content += f"### Subsection {i + 1}.{j + 1}\n\n"
            content += f"Content for subsection {i + 1}.{j + 1}.\n"
            content += f"Some detailed information here.\n\n"

    return content


def benchmark_experimental_markdown_splitter() -> None:
    """Benchmark the ExperimentalMarkdownSyntaxTextSplitter performance."""
    print("=" * 60)
    print("Markdown Text Splitter Performance Benchmark")
    print("=" * 60)

    # Test with different document sizes
    test_sizes = [10, 50, 100, 200]

    splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )

    for size in test_sizes:
        markdown_text = generate_large_markdown(size)
        doc_size_kb = len(markdown_text) / 1024

        # Warm-up run
        _ = splitter.split_text(markdown_text)

        # Measure time
        start_time = time.perf_counter()
        chunks = splitter.split_text(markdown_text)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        chunks_per_second = len(chunks) / elapsed_time if elapsed_time > 0 else 0

        print(f"\nTest Size: {size} sections")
        print(f"  Document size: {doc_size_kb:.2f} KB")
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Execution time: {elapsed_time:.4f} seconds")
        print(f"  Chunks per second: {chunks_per_second:.0f}")
        print(f"  Time per chunk: {(elapsed_time / len(chunks) * 1000):.2f} ms")

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_experimental_markdown_splitter()
