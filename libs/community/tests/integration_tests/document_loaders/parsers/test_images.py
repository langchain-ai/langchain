import re
from pathlib import Path
from typing import Any, Type

import pytest
from langchain_core.documents.base import Blob
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import ChatMessage

from langchain_community.document_loaders.parsers.images import (
    LLMImageBlobParser,
    RapidOCRBlobParser,
    TesseractBlobParser,
)

path_base = Path(__file__).parent.parent.parent
building_image = Blob.from_path(path_base / "examples/building.jpg")
text_image = Blob.from_path(path_base / "examples/text.png")
page_image = Blob.from_path(path_base / "examples/page.png")

_re_in_image = r"(?ms).*MAKE.*TEXT.*STAND.*OUT.*FROM.*"


@pytest.mark.parametrize(
    "blob,body",
    [
        # (building_image, ""),
        # (Blob.from_path(path_base / "examples/text.png"), _re_in_image),
        (Blob.from_path(path_base / "examples/text-gray.png"), _re_in_image),
        # (Blob.from_path(path_base / "examples/sample-bw.png"), _re_in_image),
        # # (Blob.from_path(path_base / "examples/sample-color.png"), _re_in_image),
        # # (Blob.from_path(path_base / "examples/sample-gray.png"), _re_in_image),
    ],
)
@pytest.mark.parametrize(
    "blob_loader,kw",
    [
        (RapidOCRBlobParser, {}),
        (TesseractBlobParser, {}),
        (
            LLMImageBlobParser,
            {
                "model": FakeMessagesListChatModel(
                    responses=[
                        ChatMessage(
                            id="ai1",
                            role="system",
                            content="A building. MAKE TEXT STAND  OUT FROM BACKGROUNDS",
                        ),
                    ]
                )
            },
        ),
    ],
)
def test_image_parser_with_differents_files(
    blob_loader: Type,
    kw: dict[str, Any],
    blob: Blob,
    body: str,
) -> None:
    if blob_loader == LLMImageBlobParser and "building" in str(blob.path):
        body = ".*building.*"
    documents = list(blob_loader(**kw).lazy_parse(blob))
    assert len(documents) == 1
    assert re.compile(body).match(documents[0].page_content)


@pytest.mark.parametrize(
    "blob_loader,kw",
    [
        (RapidOCRBlobParser, {}),
        (TesseractBlobParser, {}),
        (
            LLMImageBlobParser,
            {
                "model": FakeMessagesListChatModel(
                    responses=[
                        ChatMessage(
                            id="ai1",
                            role="system",
                            content="A building. MAKE TEXT STAND  OUT FROM BACKGROUNDS",
                        ),
                    ]
                )
            },
        ),
    ],
)
def test_image_parser_with_numpy(
    blob_loader: Type,
    kw: dict[str, Any],
) -> None:
    blob = Blob.from_path(
        path_base / "examples/gray.npy", mime_type="application/x-npy"
    )
    documents = list(blob_loader(**kw).lazy_parse(blob))
    assert len(documents) == 1
