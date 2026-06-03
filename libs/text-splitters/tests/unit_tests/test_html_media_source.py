"""Tests for HTMLSemanticPreservingSplitter nested source tag support."""

from langchain_text_splitters.html import HTMLSemanticPreservingSplitter


def test_preserve_video_with_nested_source():
    """Video tag with nested <source> should preserve the src URL."""
    html = """
    <h1>Media</h1>
    <video controls>
        <source src="https://example.com/video.mp4" type="video/mp4" />
    </video>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[("h1", "Header 1")],
        preserve_videos=True,
    )
    docs = splitter.split_text(html)
    assert "https://example.com/video.mp4" in docs[0].page_content


def test_preserve_audio_with_nested_source():
    """Audio tag with nested <source> should preserve the src URL."""
    html = """
    <h1>Media</h1>
    <audio controls>
        <source src="https://example.com/audio.mp3" type="audio/mpeg" />
    </audio>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[("h1", "Header 1")],
        preserve_audio=True,
    )
    docs = splitter.split_text(html)
    assert "https://example.com/audio.mp3" in docs[0].page_content


def test_preserve_video_direct_src_still_works():
    """Video tag with direct src attribute should continue to work."""
    html = """
    <h1>Media</h1>
    <video src="https://example.com/direct.mp4" controls></video>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[("h1", "Header 1")],
        preserve_videos=True,
    )
    docs = splitter.split_text(html)
    assert "https://example.com/direct.mp4" in docs[0].page_content


def test_preserve_audio_direct_src_still_works():
    """Audio tag with direct src attribute should continue to work."""
    html = """
    <h1>Media</h1>
    <audio src="https://example.com/direct.mp3" controls></audio>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[("h1", "Header 1")],
        preserve_audio=True,
    )
    docs = splitter.split_text(html)
    assert "https://example.com/direct.mp3" in docs[0].page_content


def test_preserve_video_no_src_emtpy_link():
    """Video tag without any src should still produce a placeholder."""
    html = """
    <h1>Media</h1>
    <video controls></video>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[("h1", "Header 1")],
        preserve_videos=True,
    )
    docs = splitter.split_text(html)
    # Should contain empty video marker
    assert "![video:]" in docs[0].page_content
