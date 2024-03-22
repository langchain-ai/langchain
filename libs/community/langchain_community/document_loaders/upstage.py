"""Upstage document loader using Layout-Analyzer Model."""
from enum import Enum
from typing import List

import requests
import os
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class OutputType(Enum):
    TEXT = "text"
    HTML = "html"


class SplitType(Enum):
    NONE = "none"
    PAGE = "page"
    ELEMENT = "element"


def validate_api_key(api_key: str):
    if not api_key:
        raise ValueError("API Key is required for Upstage Document Loader")


def validate_file_path(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def parse_output(data: dict, output_type: OutputType):
    if (output_type) == OutputType.TEXT:
        return data["text"]
    elif (output_type) == OutputType.HTML:
        return data["text"]


LAYOUT_ANALYZER_URL = "https://api.upstage.ai/v1/document-ai/layout-analyzer"


class UpstageDocumentLoader(BaseLoader):

    def __init__(
        self,
        file_path: str,
        output_type: OutputType = OutputType.TEXT,
        split: SplitType = SplitType.NONE,
        api_key: str = ""
    ):
        """Initialize with the file path."""
        self.file_path = file_path
        self.output_type = output_type
        self.split = split
        self.api_key = api_key
        self.file_name = os.path.basename(file_path)

        validate_file_path(self.file_path)
        validate_api_key(self.api_key)

    def load(self) -> List[Document]:
        """Load data into Document objects."""

        url = LAYOUT_ANALYZER_URL
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"document": open(self.file_name, "rb")}
        response = requests.post(url, headers=headers, files=files)

        if (response.status_code != 200):
            raise ValueError(f"API call error: {response.status_code}")

        json = response.json()

        if (self.split) == SplitType.NONE:
            return [Document(page_content=(parse_output(json, self.output_type)), metadata={"total_pages": json["billed_pages"], "type": self.output_type, "split": self.split})]
        elif (self.split) == SplitType.ELEMENT:
            for element in json["elements"]:
                yield Document(page_content=(parse_output(element, self.output_type)), metadata={"page": element["page"], "id": element["id"], "type": self.output_type, "split": self.split})
        elif (self.split) == SplitType.PAGE:
            elements = json["elements"]
            pages = set(map(lambda x: x["page"], elements))
            pages.sort()

            page_group = [
                [element for element in elements if element["page"] == x] for x in pages]

            for group in page_group:
                page_content = ""
                for element in group:
                    page_content += parse_output(element,
                                                 self.output_type) + " "
                yield Document(page_content=page_content.strip(), metadata={"page": group[0]["page"], "type": self.output_type, "split": self.split})

        return []
