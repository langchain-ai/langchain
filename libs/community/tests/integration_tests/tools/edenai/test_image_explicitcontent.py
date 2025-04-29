"""Test EdenAi's image moderation Tool .

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""

from langchain_community.tools.edenai import EdenAiExplicitImageTool


def test_edenai_call() -> None:
    """Test simple call to edenai's image moderation endpoint."""
    image_moderation = EdenAiExplicitImageTool(providers=["amazon"])

    output = image_moderation.invoke("https://static.javatpoint.com/images/objects.jpg")

    assert image_moderation.name == "edenai_image_explicit_content_detection"
    assert image_moderation.feature == "image"
    assert image_moderation.subfeature == "explicit_content"
    assert isinstance(output, str)
