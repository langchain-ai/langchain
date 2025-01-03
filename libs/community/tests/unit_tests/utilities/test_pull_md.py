import pytest

from utilities.pull_md import PullMdAPIWrapper


def test_convert_url_to_markdown_success(mocker):
    """Test successful URL conversion to Markdown."""
    expected_markdown = "# Example Domain"
    mocker.patch('pull_md.pull_markdown', return_value=expected_markdown)

    pull_md = PullMdAPIWrapper()
    result = pull_md.convert_url_to_markdown("http://example.com")
    assert result == expected_markdown


def test_convert_url_to_markdown_failure(mocker):
    """Test failure in URL conversion to Markdown."""
    mocker.patch('pull_md.pull_markdown', side_effect=Exception("Failed to convert"))

    pull_md = PullMdAPIWrapper()
    with pytest.raises(Exception) as excinfo:
        pull_md.convert_url_to_markdown("http://example.com")
    assert "Failed to convert" in str(excinfo.value)
