import re
from pathlib import Path
from typing import Any

import pytest

from langchain_community.document_loaders.parsers.images import RapidOCRBlobParser, \
    _ImageBlobParser, TesseractBlobParser, MultimodalBlobParser
from langchain_core.documents.base import Blob
from langchain_openai import ChatOpenAI

building_image = Blob.from_path(Path(__file__).parent.parent / "examples/building.jpg")
text_image = Blob.from_path(Path(__file__).parent.parent / "examples/text.png")


@pytest.mark.parametrize(
    'blob,body',
    [
        (building_image, ""),
        (text_image, r".*\bMAKE *TEXT\b.*\bSTAND\b.*\bOUT *FROM\b.*\bBACKGROUNDS\b.*")
    ]
)
@pytest.mark.parametrize(
    "format,pattern",
    [
        ("text", r"(?sm)^{body}$"),
        ("markdown", r"(?sm)^!\[{body}]\(\.\)|$"),
        ("html", r'(?sm)^(<img alt="{body}" />|)'),
    ],
)
@pytest.mark.parametrize(
    "blob_loader,kw",
    [
        (RapidOCRBlobParser,{}),
        (TesseractBlobParser,{}),
        (MultimodalBlobParser,{"model":ChatOpenAI(model="gpt-4o", max_tokens=1024)})
    ]
)
def test_image_parser_with_differents_format_and_files(
        blob_loader, #: _ImageBlobParser,
        kw:dict[str,any],
        blob: Blob,
        body: str,
        format: str,
        pattern: str,
    ) -> None:
    if blob_loader == MultimodalBlobParser:
        body=".*building.*"
    documents = list(blob_loader(format=format,**kw).lazy_parse(blob))
    assert (len(documents) == 1)
    assert re.compile(pattern.format(body=body)).match(documents[0].page_content)
