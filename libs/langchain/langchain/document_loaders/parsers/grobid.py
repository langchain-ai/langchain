import logging
from typing import Dict, Iterator, List, Union

import requests
from langchain_core.documents import Document

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob

logger = logging.getLogger(__name__)


class ServerUnavailableException(Exception):
    """Exception raised when the Grobid server is unavailable."""

    pass


class GrobidParser(BaseBlobParser):
    """Load  article `PDF` files using `Grobid`."""

    def __init__(
        self,
        segment_sentences: bool,
        grobid_server: str = "http://localhost:8070/api/processFulltextDocument",
    ) -> None:
        self.segment_sentences = segment_sentences
        self.grobid_server = grobid_server
        try:
            requests.get(grobid_server)
        except requests.exceptions.RequestException:
            logger.error(
                "GROBID server does not appear up and running, \
                please ensure Grobid is installed and the server is running"
            )
            raise ServerUnavailableException

    def process_xml(
        self, file_path: str, xml_data: str, segment_sentences: bool
    ) -> Iterator[Document]:
        """Process the XML file from Grobin."""

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "`bs4` package not found, please install it with " "`pip install bs4`"
            )
        soup = BeautifulSoup(xml_data, "xml")
        sections = soup.find_all("div")
        title = soup.find_all("title")[0].text
        chunks = []
        for section in sections:
            sect = section.find("head")
            if sect is not None:
                for i, paragraph in enumerate(section.find_all("p")):
                    chunk_bboxes = []
                    paragraph_text = []
                    for i, sentence in enumerate(paragraph.find_all("s")):
                        paragraph_text.append(sentence.text)
                        sbboxes = []
                        for bbox in sentence.get("coords").split(";"):
                            box = bbox.split(",")
                            sbboxes.append(
                                {
                                    "page": box[0],
                                    "x": box[1],
                                    "y": box[2],
                                    "h": box[3],
                                    "w": box[4],
                                }
                            )
                        chunk_bboxes.append(sbboxes)
                        if segment_sentences is True:
                            fpage, lpage = sbboxes[0]["page"], sbboxes[-1]["page"]
                            sentence_dict = {
                                "text": sentence.text,
                                "para": str(i),
                                "bboxes": [sbboxes],
                                "section_title": sect.text,
                                "section_number": sect.get("n"),
                                "pages": (fpage, lpage),
                            }
                            chunks.append(sentence_dict)
                    if segment_sentences is not True:
                        fpage, lpage = (
                            chunk_bboxes[0][0]["page"],
                            chunk_bboxes[-1][-1]["page"],
                        )
                        paragraph_dict = {
                            "text": "".join(paragraph_text),
                            "para": str(i),
                            "bboxes": chunk_bboxes,
                            "section_title": sect.text,
                            "section_number": sect.get("n"),
                            "pages": (fpage, lpage),
                        }
                        chunks.append(paragraph_dict)

        yield from [
            Document(
                page_content=chunk["text"],
                metadata=dict(
                    {
                        "text": str(chunk["text"]),
                        "para": str(chunk["para"]),
                        "bboxes": str(chunk["bboxes"]),
                        "pages": str(chunk["pages"]),
                        "section_title": str(chunk["section_title"]),
                        "section_number": str(chunk["section_number"]),
                        "paper_title": str(title),
                        "file_path": str(file_path),
                    }
                ),
            )
            for chunk in chunks
        ]

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        file_path = blob.source
        if file_path is None:
            raise ValueError("blob.source cannot be None.")
        pdf = open(file_path, "rb")
        files = {"input": (file_path, pdf, "application/pdf", {"Expires": "0"})}
        try:
            data: Dict[str, Union[str, List[str]]] = {}
            for param in ["generateIDs", "consolidateHeader", "segmentSentences"]:
                data[param] = "1"
            data["teiCoordinates"] = ["head", "s"]
            files = files or {}
            r = requests.request(
                "POST",
                self.grobid_server,
                headers=None,
                params=None,
                files=files,
                data=data,
                timeout=60,
            )
            xml_data = r.text
        except requests.exceptions.ReadTimeout:
            logger.error("GROBID server timed out. Return None.")
            xml_data = None

        if xml_data is None:
            return iter([])
        else:
            return self.process_xml(file_path, xml_data, self.segment_sentences)
