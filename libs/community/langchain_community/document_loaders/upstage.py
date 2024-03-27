import os
from enum import Enum
from typing import List

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

LAYOUT_ANALYZER_URL = "https://api.upstage.ai/v1/document-ai/layout-analyzer"


class OutputType(Enum):
    """
    Represents the output type for a document loader.
    """

    TEXT = "text"
    HTML = "html"


class SplitType(Enum):
    """
    Enum class representing the type of split for a document.

    Attributes:
        NONE (str): Represents no split.
        ELEMENT (str): Represents splitting by element.
        PAGE (str): Represents splitting by page.
    """

    NONE = "none"
    ELEMENT = "element"
    PAGE = "page"


def validate_api_key(api_key: str) -> None:
    """
    Validates the provided API key.

    Args:
        api_key (str): The API key to be validated.

    Raises:
        ValueError: If the API key is empty or None.

    Returns:
        None
    """
    if not api_key:
        raise ValueError("API Key is required for Upstage Document Loader")


def validate_file_path(file_path: str) -> None:
    """
    Validates if a file exists at the given file path.

    Args:
        file_path (str): The path to the file.

    Raises:
        FileNotFoundError: If the file does not exist at the given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def parse_output(data: dict, output_type: str) -> str:
    """
    Parse the output data based on the specified output type.

    Args:
        data (dict): The data to be parsed.
        output_type (str): The type of output to parse.

    Returns:
        str: The parsed output.

    Raises:
        ValueError: If the output type is invalid.
    """
    if (output_type) == OutputType.TEXT.value:
        return data["text"]
    elif (output_type) == OutputType.HTML.value:
        return data["html"]
    else:
        raise ValueError(f"Invalid output type: {output_type}")


class UpstageDocumentLoader(BaseLoader):
    """
    A document loader for Upstage API.

    Args:
        file_path (str): The path to the file to be loaded.
        output_type (str, optional): The desired output type.
                                     Defaults to OutputType.TEXT.value.
        split (str, optional): The split type for the document.
                               Defaults to SplitType.NONE.value.
        api_key (str, optional): The API key for authentication. Defaults to "".

    Attributes:
        file_path (str): The path to the file to be loaded.
        output_type (str): The desired output type.
        split (str): The split type for the document.
        api_key (str): The API key for authentication.
        file_name (str): The name of the file.

    Raises:
        ValueError: If the API call returns a non-200 status code.
        ValueError: If an invalid split type is provided.

    """

    def __init__(
        self,
        file_path: str,
        output_type: str = OutputType.TEXT.value,
        split: str = SplitType.NONE.value,
        api_key: str = "",
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
        """Load data into Document objects.

        This method sends a POST request to the LAYOUT_ANALYZER_URL
        with the provided API key and file. It parses the response
        JSON and creates Document objects based on the split type.

        Returns:
            List[Document]: A list of Document objects containing the loaded data.

        Raises:
            ValueError: If the API call returns a non-200 status code
                        or an invalid split type is provided.
        """

        url = LAYOUT_ANALYZER_URL
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"document": open(self.file_path, "rb")}
        response = requests.post(url, headers=headers, files=files)

        if response.status_code != 200:
            raise ValueError(f"API call error: {response.status_code}")

        json = response.json()

        if (self.split) == SplitType.NONE.value:
            # Split by document (NONE)
            docs = []
            docs.append(
                Document(
                    page_content=(parse_output(json, self.output_type)),
                    metadata={
                        "total_pages": json["billed_pages"],
                        "type": self.output_type,
                        "split": self.split,
                    },
                )
            )
            return docs

        elif (self.split) == SplitType.ELEMENT.value:
            # Split by element
            docs = []
            for element in json["elements"]:
                docs.append(
                    Document(
                        page_content=(parse_output(element, self.output_type)),
                        metadata={
                            "page": element["page"],
                            "id": element["id"],
                            "type": self.output_type,
                            "split": self.split,
                        },
                    )
                )

            return docs

        elif (self.split) == SplitType.PAGE.value:
            # Split by page
            elements = json["elements"]
            pages = sorted(set(map(lambda x: x["page"], elements)))

            page_group = [
                [element for element in elements if element["page"] == x] for x in pages
            ]

            docs = []
            for group in page_group:
                page_content = ""
                for element in group:
                    page_content += parse_output(element, self.output_type) + " "
                docs.append(
                    Document(
                        page_content=page_content.strip(),
                        metadata={
                            "page": group[0]["page"],
                            "type": self.output_type,
                            "split": self.split,
                        },
                    )
                )

            return docs

        else:
            # Invalid split type
            raise ValueError(f"Invalid split type: {self.split}")

        return []
