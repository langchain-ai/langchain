import re
from pathlib import Path
from typing import Any, Type

import pytest
from langchain_core.documents.base import Blob
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import ChatMessage

from langchain_community.document_loaders.parsers.images import (
    MultimodalBlobParser,
    RapidOCRBlobParser,
    TesseractBlobParser,
)

path_base = Path(__file__).parent.parent.parent
building_image = Blob.from_path(path_base / "examples/building.jpg")
text_image = Blob.from_path(path_base / "examples/text.png")


@pytest.mark.parametrize(
    "blob,body",
    [
        (building_image, ""),
        (text_image, r".*\bMAKE *TEXT\b.*\bSTAND\b.*\bOUT *FROM\b.*\bBACKGROUNDS\b.*"),
    ],
)
@pytest.mark.parametrize(
    "format,pattern",
    [
        ("text", r"(?ism)^{body}$"),
        ("markdown", r"(?ism)^!\[{body}]\(\.\)|$"),
        ("html", r'(?ism)^(<img alt="{body}" />|)'),
    ],
)
@pytest.mark.parametrize(
    "blob_loader,kw",
    [
        (RapidOCRBlobParser, {}),
        (TesseractBlobParser, {}),
        (
            MultimodalBlobParser,
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
def test_image_parser_with_differents_format_and_files(
    blob_loader: Type,
    kw: dict[str, Any],
    format: str,
    pattern: str,
    blob: Blob,
    body: str,
) -> None:
    if blob_loader == MultimodalBlobParser and "building" in str(blob.path):
        body = ".*building.*"
    documents = list(blob_loader(format=format, **kw).lazy_parse(blob))
    assert len(documents) == 1
    assert re.compile(pattern.format(body=body)).match(documents[0].page_content)
