"""Test EdenAi's invoice parser Tool .

In order to run this test, you need to have an EdenAI api key.
You can get it by registering for free at https://app.edenai.run/user/register.
A test key can be found at https://app.edenai.run/admin/account/settings by
clicking on the 'sandbox' toggle.
(calls will be free, and will return dummy results)

You'll then need to set EDENAI_API_KEY environment variable to your api key.
"""

from langchain_community.tools.edenai import EdenAiParsingInvoiceTool


def test_edenai_call() -> None:
    """Test simple call to edenai's invoice parser endpoint."""
    invoice_parser = EdenAiParsingInvoiceTool(providers=["amazon"], language="en")  # type: ignore[call-arg]

    output = invoice_parser.invoke(
        "https://app.edenai.run/assets/img/data_1.72e3bdcc.png"
    )

    assert invoice_parser.name == "edenai_invoice_parsing"
    assert invoice_parser.feature == "ocr"
    assert invoice_parser.subfeature == "invoice_parser"
    assert isinstance(output, str)
