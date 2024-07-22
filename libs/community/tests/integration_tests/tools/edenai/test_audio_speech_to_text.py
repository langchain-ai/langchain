"""Test EdenAi's speech to text Tool .

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""

from langchain_community.tools.edenai import EdenAiSpeechToTextTool


def test_edenai_call() -> None:
    """Test simple call to edenai's speech to text endpoint."""
    speech2text = EdenAiSpeechToTextTool(providers=["amazon"])  # type: ignore[call-arg]

    output = speech2text.invoke(
        "https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3"
    )

    assert speech2text.name == "edenai_speech_to_text"
    assert speech2text.feature == "audio"
    assert speech2text.subfeature == "speech_to_text_async"
    assert isinstance(output, str)
