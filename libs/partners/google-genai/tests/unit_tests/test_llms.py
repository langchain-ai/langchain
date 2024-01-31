from langchain_google_genai.llms import GoogleModelFamily


def test_model_family() -> None:
    model = GoogleModelFamily("gemini-pro")
    assert model == GoogleModelFamily.GEMINI
    model = GoogleModelFamily("gemini-ultra")
    assert model == GoogleModelFamily.GEMINI
