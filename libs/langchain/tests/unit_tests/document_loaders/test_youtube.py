import pytest

from langchain.document_loaders import YoutubeLoader


@pytest.mark.parametrize(
    "youtube_url, expected_video_id",
    [
        ("http://www.youtube.com/watch?v=-wtIMTCHWuI", "-wtIMTCHWuI"),
        ("http://youtube.com/watch?v=-wtIMTCHWuI", "-wtIMTCHWuI"),
        ("http://m.youtube.com/watch?v=-wtIMTCHWuI", "-wtIMTCHWuI"),
        ("http://youtu.be/-wtIMTCHWuI", "-wtIMTCHWuI"),
        ("https://youtu.be/-wtIMTCHWuI", "-wtIMTCHWuI"),
        ("https://www.youtube.com/watch?v=lalOy8Mbfdc", "lalOy8Mbfdc"),
        ("https://m.youtube.com/watch?v=lalOy8Mbfdc", "lalOy8Mbfdc"),
        ("https://youtube.com/watch?v=lalOy8Mbfdc", "lalOy8Mbfdc"),
        ("http://youtu.be/lalOy8Mbfdc?t=1", "lalOy8Mbfdc"),
        ("http://youtu.be/lalOy8Mbfdc?t=1s", "lalOy8Mbfdc"),
        ("https://youtu.be/lalOy8Mbfdc?t=1", "lalOy8Mbfdc"),
        ("http://www.youtube-nocookie.com/embed/lalOy8Mbfdc?rel=0", "lalOy8Mbfdc"),
        ("https://youtu.be/lalOy8Mbfdc?t=1s", "lalOy8Mbfdc"),
        ("https://www.youtube.com/shorts/cd0Fy92_w_s", "cd0Fy92_w_s"),
    ],
)
def test_video_id_extraction(youtube_url: str, expected_video_id: str) -> None:
    """Test that the video id is extracted from a youtube url"""
    assert YoutubeLoader.extract_video_id(youtube_url) == expected_video_id
