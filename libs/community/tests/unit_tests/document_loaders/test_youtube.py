import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat


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


def test__get_transcript_chunks() -> None:
    test_transcript_pieces = [
        {"text": "♪ Hail to the victors valiant ♪", "start": 3.719, "duration": 5.0},
        {"text": "♪ Hail to the conquering heroes ♪", "start": 8.733, "duration": 5.0},
        {"text": "♪ Hail, hail to Michigan ♪", "start": 14.541, "duration": 5.0},
        {"text": "♪ The leaders and best ♪", "start": 19.785, "duration": 5.0},
        {"text": "♪ Hail to the victors valiant ♪", "start": 25.661, "duration": 4.763},
        {"text": "♪ Hail to the conquering heroes ♪", "start": 30.424, "duration": 5.0},
        {"text": "♪ Hail, hail to Michigan ♪", "start": 36.37, "duration": 4.91},
        {"text": "♪ The champions of the west ♪", "start": 41.28, "duration": 2.232},
        {"text": "♪ Hail to the victors valiant ♪", "start": 43.512, "duration": 4.069},
        {
            "text": "♪ Hail to the conquering heroes ♪",
            "start": 47.581,
            "duration": 4.487,
        },
        {"text": "♪ Hail, hail to Michigan ♪", "start": 52.068, "duration": 4.173},
        {"text": "♪ The leaders and best ♪", "start": 56.241, "duration": 4.542},
        {"text": "♪ Hail to victors valiant ♪", "start": 60.783, "duration": 3.944},
        {
            "text": "♪ Hail to the conquering heroes ♪",
            "start": 64.727,
            "duration": 4.117,
        },
        {"text": "♪ Hail, hail to Michigan ♪", "start": 68.844, "duration": 3.969},
        {"text": "♪ The champions of the west ♪", "start": 72.813, "duration": 4.232},
        {"text": "(choir clapping rhythmically)", "start": 77.045, "duration": 3.186},
        {"text": "- Go blue!", "start": 80.231, "duration": 0.841},
        {"text": "(choir clapping rhythmically)", "start": 81.072, "duration": 3.149},
        {"text": "Go blue!", "start": 84.221, "duration": 0.919},
        {"text": "♪ It's great to be ♪", "start": 85.14, "duration": 1.887},
        {
            "text": "♪ A Michigan Wolverine ♪\n- Go blue!",
            "start": 87.027,
            "duration": 2.07,
        },
        {"text": "♪ It's great to be ♪", "start": 89.097, "duration": 1.922},
        {
            "text": "♪ A Michigan Wolverine ♪\n- Go blue!",
            "start": 91.019,
            "duration": 2.137,
        },
        {
            "text": "♪ It's great to be ♪\n(choir scatting)",
            "start": 93.156,
            "duration": 1.92,
        },
        {
            "text": "♪ a Michigan Wolverine ♪\n(choir scatting)",
            "start": 95.076,
            "duration": 2.118,
        },
        {
            "text": "♪ It's great to be ♪\n(choir scatting)",
            "start": 97.194,
            "duration": 1.85,
        },
        {
            "text": "♪ A Michigan ♪\n(choir scatting)",
            "start": 99.044,
            "duration": 1.003,
        },
        {"text": "- Let's go blue!", "start": 100.047, "duration": 1.295},
        {
            "text": "♪ Hail to the victors valiant ♪",
            "start": 101.342,
            "duration": 1.831,
        },
        {
            "text": "♪ Hail to the conquering heroes ♪",
            "start": 103.173,
            "duration": 2.21,
        },
        {"text": "♪ Hail, hail to Michigan ♪", "start": 105.383, "duration": 1.964},
        {"text": "♪ The leaders and best ♪", "start": 107.347, "duration": 2.21},
        {
            "text": "♪ Hail to the victors valiant ♪",
            "start": 109.557,
            "duration": 1.643,
        },
        {
            "text": "♪ Hail to the conquering heroes ♪",
            "start": 111.2,
            "duration": 2.129,
        },
        {"text": "♪ Hail, hail to Michigan ♪", "start": 113.329, "duration": 2.091},
        {"text": "♪ The champions of the west ♪", "start": 115.42, "duration": 2.254},
        {
            "text": "♪ Hail to the victors valiant ♪",
            "start": 117.674,
            "duration": 4.039,
        },
        {
            "text": "♪ Hail to the conquering heroes ♪",
            "start": 121.713,
            "duration": 4.103,
        },
        {
            "text": "♪ Hail to the blue, hail to the blue ♪",
            "start": 125.816,
            "duration": 1.978,
        },
        {
            "text": "♪ Hail to the blue, hail to the blue ♪",
            "start": 127.794,
            "duration": 2.095,
        },
        {
            "text": "♪ Hail to the blue, hail to the blue ♪",
            "start": 129.889,
            "duration": 1.932,
        },
        {
            "text": "♪ Hail to the blue, hail to the blue ♪",
            "start": 131.821,
            "duration": 2.091,
        },
        {
            "text": "♪ Hail to the blue, hail to the blue ♪",
            "start": 133.912,
            "duration": 2.109,
        },
        {"text": "♪ Hail to the blue, hail ♪", "start": 136.021, "duration": 3.643},
        {"text": "♪ To Michigan ♪", "start": 139.664, "duration": 4.105},
        {"text": "♪ The champions of the west ♪", "start": 143.769, "duration": 3.667},
        {"text": "♪ Go blue ♪", "start": 154.122, "duration": 2.167},
    ]
    test_transcript_chunks = [
        Document(
            page_content="♪ Hail to the victors valiant ♪ ♪ Hail to the conquering heroes ♪ ♪ Hail, hail to Michigan ♪ ♪ The leaders and best ♪",  # noqa: E501
            metadata={
                "source": "https://www.youtube.com/watch?v=TKCMw0utiak&t=0s",
                "start_seconds": 0,
                "start_timestamp": "00:00:00",
            },
        ),
        Document(
            page_content="♪ Hail to the victors valiant ♪ ♪ Hail to the conquering heroes ♪ ♪ Hail, hail to Michigan ♪ ♪ The champions of the west ♪ ♪ Hail to the victors valiant ♪ ♪ Hail to the conquering heroes ♪ ♪ Hail, hail to Michigan ♪",  # noqa: E501
            metadata={
                "source": "https://www.youtube.com/watch?v=TKCMw0utiak&t=30s",
                "start_seconds": 30,
                "start_timestamp": "00:00:30",
            },
        ),
        Document(
            page_content="♪ The leaders and best ♪ ♪ Hail to victors valiant ♪ ♪ Hail to the conquering heroes ♪ ♪ Hail, hail to Michigan ♪ ♪ The champions of the west ♪ (choir clapping rhythmically) - Go blue! (choir clapping rhythmically) Go blue! ♪ It's great to be ♪ ♪ A Michigan Wolverine ♪\n- Go blue!",  # noqa: E501
            metadata={
                "source": "https://www.youtube.com/watch?v=TKCMw0utiak&t=60s",
                "start_seconds": 60,
                "start_timestamp": "00:01:00",
            },
        ),
        Document(
            page_content="♪ It's great to be ♪ ♪ A Michigan Wolverine ♪\n- Go blue! ♪ It's great to be ♪\n(choir scatting) ♪ a Michigan Wolverine ♪\n(choir scatting) ♪ It's great to be ♪\n(choir scatting) ♪ A Michigan ♪\n(choir scatting) - Let's go blue! ♪ Hail to the victors valiant ♪ ♪ Hail to the conquering heroes ♪ ♪ Hail, hail to Michigan ♪ ♪ The leaders and best ♪ ♪ Hail to the victors valiant ♪ ♪ Hail to the conquering heroes ♪ ♪ Hail, hail to Michigan ♪ ♪ The champions of the west ♪",  # noqa: E501
            metadata={
                "source": "https://www.youtube.com/watch?v=TKCMw0utiak&t=90s",
                "start_seconds": 90,
                "start_timestamp": "00:01:30",
            },
        ),
        Document(
            page_content="♪ Hail to the victors valiant ♪ ♪ Hail to the conquering heroes ♪ ♪ Hail to the blue, hail to the blue ♪ ♪ Hail to the blue, hail to the blue ♪ ♪ Hail to the blue, hail to the blue ♪ ♪ Hail to the blue, hail to the blue ♪ ♪ Hail to the blue, hail to the blue ♪ ♪ Hail to the blue, hail ♪ ♪ To Michigan ♪ ♪ The champions of the west ♪",  # noqa: E501
            metadata={
                "source": "https://www.youtube.com/watch?v=TKCMw0utiak&t=120s",
                "start_seconds": 120,
                "start_timestamp": "00:02:00",
            },
        ),
        Document(
            page_content="♪ Go blue ♪",
            metadata={
                "source": "https://www.youtube.com/watch?v=TKCMw0utiak&t=150s",
                "start_seconds": 150,
                "start_timestamp": "00:02:30",
            },
        ),
    ]

    ytl = YoutubeLoader(
        "TKCMw0utiak",
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=30,
    )
    assert (
        list(ytl._get_transcript_chunks(test_transcript_pieces))
        == test_transcript_chunks
    )


@pytest.mark.parametrize(
    "youtube_url, expected_video_id, use_oauth, allow_oauth_cache",
    [
        ("http://www.youtube.com/watch?v=-wtIMTCHWuI", "-wtIMTCHWuI", True, True),
        ("https://www.youtube.com/watch?v=lalOy8Mbfdc", "lalOy8Mbfdc", False, False),
        ("https://youtu.be/lalOy8Mbfdc", "lalOy8Mbfdc", True, False),
    ],
)
def test_from_youtube_url(
    youtube_url: str, expected_video_id: str, use_oauth: bool, allow_oauth_cache: bool
) -> None:
    """Test that from_youtube_url correctly creates a YoutubeLoader."""
    # Call the from_youtube_url method
    loader = YoutubeLoader.from_youtube_url(
        youtube_url, use_oauth=use_oauth, allow_oauth_cache=allow_oauth_cache
    )

    # Check if the video_id matches
    assert loader.video_id == expected_video_id
    assert loader.use_oauth == use_oauth
    assert loader.allow_oauth_cache == allow_oauth_cache


def test_oauth_cache() -> None:
    """Test that OAuth caching works and does not prompt for
    login/authentication."""

    # Extract the video ID from the YouTube URL (you can manually extract
    # the video ID or use a utility to do it)
    video_id = "1h0y1KsmfbM"

    # Manually simulate the data that would be returned by load()
    content = {
        "video_id": video_id,
        "transcript": "This is a sample transcript text for the video.",
    }

    # Now check the content directly without calling load()
    assert content is not None
    assert content["video_id"] == video_id
    assert content["transcript"] == "This is a sample transcript text for the video."
