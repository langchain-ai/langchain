import os
from pathlib import Path
from typing import List, Literal, Union

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

LAYOUT_ANALYZER_URL = "https://api.upstage.ai/v1/document-ai/layout-analyzer"

OutputType = Literal["text", "html"]
SplitType = Literal["none", "element", "page"]


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


def parse_output(data: dict, output_type: OutputType) -> str:
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
    if output_type == "text":
        return data["text"]
    elif output_type == "html":
        return data["html"]
    else:
        raise ValueError(f"Invalid output type: {output_type}")


class UpstageDocumentLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        output_type: OutputType = "text",
        split: SplitType = "none",
        api_key: str = "",
        url: str = LAYOUT_ANALYZER_URL,
    ):
        """
        Initializes an instance of the Upstage document loader.

        Args:
            file_path (Union[str, Path]): The path to the input file.
            output_type (OutputType, optional): The desired output type.
                                                Defaults to "text".
            split (SplitType, optional): The type of splitting to apply.
                                         Defaults to "none".
            api_key (str, optional): The API key for authentication.
                                     Defaults to an empty string.
            url (str, optional): The URL for the layout analyzer.
                                 Defaults to LAYOUT_ANALYZER_URL.
        """
        self.file_path = file_path
        self.output_type = output_type
        self.split = split
        self.api_key = api_key
        self.file_name = os.path.basename(file_path)
        self.url = url

        validate_file_path(self.file_path)
        validate_api_key(self.api_key)

    def _get_response(self) -> requests.Response:
        """
        Sends a POST request to the specified URL with the document file
        and returns the response.

        Returns:
            requests.Response: The response object from the API call.

        Raises:
            ValueError: If there is an error in the API call.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            files = {"document": open(self.file_path, "rb")}
            response = requests.post(self.url, headers=headers, files=files)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call error: {e}")
        finally:
            files["document"].close()

        return response.json()

    def lazy_load(self) -> List[Document]:
        """
        Lazily loads documents based on the split type.

        Returns:
            A generator that yields Document objects based on the split type.

        Raises:
            ValueError: If the split type is invalid.
        """
        response = self._get_response()

        if self.split == "none":
            yield Document(
                page_content=(parse_output(response, self.output_type)),
                metadata={
                    "total_pages": response["billed_pages"],
                    "type": self.output_type,
                    "split": self.split,
                },
            )

        elif self.split == "element":
            for element in response["elements"]:
                yield Document(
                    page_content=(parse_output(element, self.output_type)),
                    metadata={
                        "page": element["page"],
                        "id": element["id"],
                        "type": self.output_type,
                        "split": self.split,
                    },
                )

        elif self.split == "page":
            # Split by page
            elements = response["elements"]
            pages = sorted(set(map(lambda x: x["page"], elements)))

            page_group = [
                [element for element in elements if element["page"] == x] for x in pages
            ]

            for group in page_group:
                page_content = " ".join(
                    [parse_output(element, self.output_type) for element in group]
                )

                yield Document(
                    page_content=page_content,
                    metadata={
                        "page": group[0]["page"],
                        "type": self.output_type,
                        "split": self.split,
                    },
                )

        else:
            # Invalid split type
            raise ValueError(f"Invalid split type: {self.split}")

    def load(self) -> List[Document]:
        """
        Loads the documents from the response based on the specified split type.

        Returns:
            A list of Document objects representing the loaded documents.
        """
        response = self._get_response()

        if self.split == "none":
            # Split by document (NONE)
            docs = []
            docs.append(
                Document(
                    page_content=(parse_output(response, self.output_type)),
                    metadata={
                        "total_pages": response["billed_pages"],
                        "type": self.output_type,
                        "split": self.split,
                    },
                )
            )
            return docs

        elif self.split == "element":
            # Split by element
            docs = []
            for element in response["elements"]:
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

        elif self.split == "page":
            # Split by page
            elements = response["elements"]
            pages = sorted(set(map(lambda x: x["page"], elements)))

            page_group = [
                [element for element in elements if element["page"] == x] for x in pages
            ]

            docs = []
            for group in page_group:
                page_content = " ".join(
                    [parse_output(element, self.output_type) for element in group]
                )

                docs.append(
                    Document(
                        page_content=page_content,
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
