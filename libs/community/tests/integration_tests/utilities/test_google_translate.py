import pytest

from langchain_community.tools.google_translate.tool import GoogleTranslateRun


def googletrans_installed() -> bool:
    try:
        from googletrans import Translator  # noqa: F401
        return True
    except Exception as e:
        print(f"googletrans not installed, skipping test {e}")  # noqa: T201
        return False
    
@pytest.mark.skipif(not googletrans_installed(), reason="requires googletrans package")
def test_google_translate_tool() -> None:
    text = "সকল নোটিশ ও গুরুত্বপূর্ণ তথ্যসমূহ।"
    tool = GoogleTranslateRun()
    result = tool(text)
    print(result) # noqa: T201
    assert len(result.split()) > 3