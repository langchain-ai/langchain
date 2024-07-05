"""Integration test for video captioning."""

from langchain_openai import ChatOpenAI

from langchain_experimental.video_captioning.base import VideoCaptioningChain


def test_video_captioning_hard() -> None:
    """Test input that is considered hard for this chain to process."""
    URL = """
    https://ia904700.us.archive.org/22/items/any-chibes/X2Download.com
    -FXX%20USA%20%C2%ABPromo%20Noon%20-%204A%20Every%20Day%EF%BF%BD%EF
    %BF%BD%C2%BB%20November%202021%EF%BF%BD%EF%BF%BD-%281080p60%29.mp4
    """
    chain = VideoCaptioningChain(  # type: ignore[call-arg]
        llm=ChatOpenAI(
            model="gpt-4",
            max_tokens=4000,
        )
    )
    srt_content = chain.run(video_file_path=URL)

    assert (
        "mustache" in srt_content
        and "Any chives?" in srt_content
        and "How easy? A little tighter." in srt_content
        and "it's a little tight in" in srt_content
        and "every day" in srt_content
    )
