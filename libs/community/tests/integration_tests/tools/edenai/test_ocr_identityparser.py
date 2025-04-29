"""Test EdenAi's identity parser Tool .

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""

from langchain_community.tools.edenai import EdenAiParsingIDTool


def test_edenai_call() -> None:
    """Test simple call to edenai's identity parser endpoint."""
    id_parser = EdenAiParsingIDTool(providers=["amazon"], language="en")

    output = id_parser.invoke(
        "https://www.citizencard.com/images/citizencard-uk-id-card-2023.jpg"
    )

    assert id_parser.name == "edenai_identity_parsing"
    assert id_parser.feature == "ocr"
    assert id_parser.subfeature == "identity_parser"
    assert isinstance(output, str)
