"""Test ChatGoogleGenerativeAI chat model."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    ChatGoogleGenerativeAIError,
)

_MODEL = "gemini-pro"  # TODO: Use nano when it's available.
_VISION_MODEL = "gemini-pro-vision"
_B64_string = """iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAABhGlDQ1BJQ0MgUHJvZmlsZQAAeJx9kT1Iw0AcxV8/xCIVQTuIKGSoTi2IijhqFYpQIdQKrTqYXPoFTRqSFBdHwbXg4Mdi1cHFWVcHV0EQ/ABxdXFSdJES/5cUWsR4cNyPd/ced+8Af6PCVDM4DqiaZaSTCSGbWxW6XxHECPoRQ0hipj4niil4jq97+Ph6F+dZ3uf+HL1K3mSATyCeZbphEW8QT29aOud94ggrSQrxOXHMoAsSP3JddvmNc9FhP8+MGJn0PHGEWCh2sNzBrGSoxFPEUUXVKN+fdVnhvMVZrdRY6578heG8trLMdZrDSGIRSxAhQEYNZVRgIU6rRoqJNO0nPPxDjl8kl0yuMhg5FlCFCsnxg//B727NwuSEmxROAF0vtv0xCnTvAs26bX8f23bzBAg8A1da219tADOfpNfbWvQI6NsGLq7bmrwHXO4Ag0+6ZEiOFKDpLxSA9zP6phwwcAv0rLm9tfZx+gBkqKvUDXBwCIwVKXvd492hzt7+PdPq7wdzbXKn5swsVgAAA8lJREFUeJx90dtPHHUUB/Dz+81vZhb2wrDI3soUKBSRcisF21iqqCRNY01NTE0k8aHpi0k18VJfjOFvUF9M44MmGrHFQqSQiKSmFloL5c4CXW6Fhb0vO3ufvczMzweiBGI9+eW8ffI95/yQqqrwv4UxBgCfJ9w/2NfSVB+Nyn6/r+vdLo7H6FkYY6yoABR2PJujj34MSo/d/nHeVLYbydmIp/bEO0fEy/+NMcbTU4/j4Vs6Lr0ccKeYuUKWS4ABVCVHmRdszbfvTgfjR8kz5Jjs+9RREl9Zy2lbVK9wU3/kWLJLCXnqza1bfVe7b9jLbIeTMcYu13Jg/aMiPrCwVFcgtDiMhnxwJ/zXVDwSdVCVMRV7nqzl2i9e/fKrw8mqSp84e2sFj3Oj8/SrF/MaicmyYhAaXu58NPAbeAeyzY0NLecmh2+ODN3BewYBAkAY43giI3kebrnsRmvV9z2D4ciOa3EBAf31Tp9sMgdxMTFm6j74/Ogb70VCYQKAAIDCXkOAIC6pkYBWdwwnpHEdf6L9dJtJKPh95DZhzFKMEWRAGL927XpWTmMA+s8DAOBYAoR483l/iHZ/8bXoODl8b9UfyH72SXepzbyRJNvjFGHKMlhvMBze+cH9+4lEuOOlU2X1tVkFTU7Om03q080NDGXV1cflRpHwaaoiiiildB8jhDLZ7HDfz2Yidba6Vn2L4fhzFrNRKy5OZ2QOZ1U5W8VtqlVH/iUHcM933zZYWS7Wtj66zZr65bzGJQt0glHgudi9XVzEl4vKw2kUPhO020oPYI1qYc+2Xc0bRXFwTLY0VXa2VibD/lBaIXm1UChN5JSRUcQQ1Tk/47Cf3x8bY7y17Y17PVYTG1UkLPBFcqik7Zoa9JcLYoHBqHhXNgd6gS1k9EJ1TQ2l9EDy1saErmQ2kGpwGC2MLOtCM8nZEV1K0tKJtEksSm26J/rHg2zzmabKisq939nHzqUH7efzd4f/nPGW6NP8ybNFrOsWQhpoCuuhnJ4hAnPhFam01K4oQMjBg/mzBjVhuvw2O++KKT+BIVxJKzQECBDLF2qu2WTMmCovtDQ1f8iyoGkUADBCCGPsdnvTW2OtFm01VeB06msvdWlpPZU0wJRG85ns84umU3k+VyxeEcWqvYUBAGsUrbvme4be99HFeisP/pwUOIZaOqQX31ISgrKmZhLHtXNXuJq68orrr5/9mBCglCLAGGPyy81votEbcjlKLrC9E8mhH3wdHRdcyyvjidSlxjftPJpD+o25JYvRHGFoZDdks1mBQhxJu9uxvwEiXuHnHbLd1AAAAABJRU5ErkJggg=="""  # noqa: E501


def test_chat_google_genai_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_chat_google_genai_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_chat_google_genai_abatch() -> None:
    """Test streaming tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_chat_google_genai_abatch_tags() -> None:
    """Test batch tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_chat_google_genai_batch() -> None:
    """Test batch tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_chat_google_genai_ainvoke() -> None:
    """Test invoke tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_chat_google_genai_invoke() -> None:
    """Test invoke tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = llm.invoke(
        "I'm Pickle Rick",
        config=dict(tags=["foo"]),
        generation_config=dict(top_k=2, top_p=1, temperature=0.7),
    )
    assert isinstance(result.content, str)
    assert not result.content.startswith(" ")


def test_chat_google_genai_invoke_multimodal() -> None:
    messages: list = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Guess what's in this picture! You have 3 guesses.",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64," + _B64_string,
                },
            ]
        ),
    ]
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL)
    response = llm.invoke(messages)
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0

    # Try streaming
    for chunk in llm.stream(messages):
        print(chunk)
        assert isinstance(chunk.content, str)
        assert len(chunk.content.strip()) > 0


def test_chat_google_genai_invoke_multimodal_too_many_messages() -> None:
    # Only supports 1 turn...
    messages: list = [
        HumanMessage(content="Hi there"),
        AIMessage(content="Hi, how are you?"),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "I'm doing great! Guess what's in this picture!",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64," + _B64_string,
                },
            ]
        ),
    ]
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL)
    with pytest.raises(ChatGoogleGenerativeAIError):
        llm.invoke(messages)


def test_chat_google_genai_invoke_multimodal_invalid_model() -> None:
    # need the vision model to support this.
    messages: list = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "I'm doing great! Guess what's in this picture!",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64," + _B64_string,
                },
            ]
        ),
    ]
    llm = ChatGoogleGenerativeAI(model=_MODEL)
    with pytest.raises(ChatGoogleGenerativeAIError):
        llm.invoke(messages)
