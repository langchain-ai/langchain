"""Test EdenAi's text to speech Tool .

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""

from urllib.parse import urlparse

from langchain_community.tools.edenai import EdenAiTextToSpeechTool


def test_edenai_call() -> None:
    """Test simple call to edenai's text to speech endpoint."""
    text2speech = EdenAiTextToSpeechTool(  # type: ignore[call-arg]
        providers=["amazon"], language="en", voice="MALE"
    )

    output = text2speech.invoke("hello")
    parsed_url = urlparse(output)

    assert text2speech.name == "edenai_text_to_speech"
    assert text2speech.feature == "audio"
    assert text2speech.subfeature == "text_to_speech"
    assert isinstance(output, str)
    assert parsed_url.scheme in ["http", "https"]
