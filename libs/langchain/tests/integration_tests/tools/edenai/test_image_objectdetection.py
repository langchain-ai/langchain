"""Test EdenAi's object detection Tool .

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""
from langchain.tools.edenai import EdenAiObjectDetectionTool


def test_edenai_call() -> None:
    """Test simple call to edenai's object detection endpoint."""
    object_detection = EdenAiObjectDetectionTool(providers=["google"])

    output = object_detection("https://static.javatpoint.com/images/objects.jpg")

    assert object_detection.name == "edenai_object_detection"
    assert object_detection.feature == "image"
    assert object_detection.subfeature == "object_detection"
    assert isinstance(output, str)
