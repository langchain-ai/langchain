"""Test EdenAi's text moderation Tool .

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""
from langchain.tools.edenai.text_moderation import EdenAiTextModerationTool


def test_edenai_call() -> None:
    """Test simple call to edenai's text moderation endpoint."""

    text_moderation = EdenAiTextModerationTool(providers=["openai"], language="en")

    output = text_moderation("i hate you")

    assert text_moderation.name == "edenai_explicit_content_detection_text"
    assert text_moderation.feature == "text"
    assert text_moderation.subfeature == "moderation"
    assert isinstance(output, str)
