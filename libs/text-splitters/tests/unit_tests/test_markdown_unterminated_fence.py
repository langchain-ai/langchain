"""Regression test: an unterminated markdown code fence must not silently drop content.

ExperimentalMarkdownSyntaxTextSplitter._resolve_code_chunk consumes every remaining line
searching for a closing fence; when none exists it returned "", so split_text dropped the
code block AND all following sections/bodies with no error. Common with truncated LLM output.

  with the fix -> PASS (following content retained)
  without it   -> FAIL (everything after the opening fence is gone)
"""
from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter


def test_unterminated_code_fence_preserves_following_content():
    text = (
        "# Title\nintro\n\n```python\nx = 1\n\n"
        "## Important Section\ncritical content\n"
    )
    docs = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False).split_text(text)
    joined = "\n".join(d.page_content for d in docs)
    assert "critical content" in joined, joined
    assert "x = 1" in joined, joined
