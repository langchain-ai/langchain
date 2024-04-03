import io
import os
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Union

import fitz
import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

LAYOUT_ANALYZER_URL = "https://api.upstage.ai/v1/document-ai/layout-analyzer"
LIMIT_OF_PAGE_REQUEST = 10

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


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


class UpstageDocumentLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        output_type: OutputType = "text",
        split: SplitType = "none",
        api_key: str = None,
        api_base: str = None,
    ):
        """
        Initializes an instance of the Upstage document loader.

        Args:
            file_path (Union[str, Path]): The path to the file to be loaded.
            output_type (OutputType, optional): The desired output type.
                                                Defaults to "text".
            split (SplitType, optional): The type of splitting to be applied.
                                         Defaults to "none".
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None.
            api_base (str, optional): The base URL for the Upstage API.
                                      Defaults to None.
        """
        self.file_path = file_path
        self.output_type = output_type
        self.split = split
        self.file_name = os.path.basename(file_path)
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY", api_key, "UPSTAGE_API_KEY"
        )
        self.api_base = get_from_param_or_env(
            "UPSTAGE_API_BASE", api_base, "UPSTAGE_API_BASE", LAYOUT_ANALYZER_URL
        )

        validate_file_path(self.file_path)
        validate_api_key(self.api_key)

    def _get_response(self, files) -> Dict:
        """
        Sends a POST request to the API endpoint with the provided files and
        returns the response.

        Args:
            files (dict): A dictionary containing the files to be sent in the request.

        Returns:
            dict: The JSON response from the API.

        Raises:
            ValueError: If there is an error in the API call.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(self.api_base, headers=headers, files=files)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call error: {e}")
        finally:
            files["document"].close()

        return response.json()

    def _split_and_request(
        self, full_docs, start_page: int, split_pages: int = LIMIT_OF_PAGE_REQUEST
    ) -> Dict:
        """
        Splits the full pdf document into partial pages and sends a request to the
        server.

        Args:
            full_docs (str): The full document to be split and requested.
            start_page (int): The starting page number for splitting the document.
            split_pages (int, optional): The number of pages to split the document into.
                                         Defaults to LIMIT_OF_PAGE_REQUEST.

        Returns:
            response: The response from the server.
        """
        with fitz.open() as chunk_pdf:
            chunk_pdf.insert_pdf(
                full_docs,
                from_page=start_page,
                to_page=start_page + split_pages - 1,
            )
            pdf_bytes = chunk_pdf.write()

        files = {"document": io.BytesIO(pdf_bytes)}
        response = self._get_response(files)

        return response

    def _element_document(self, element) -> List[Document]:
        """
        Converts an element into a Document object.

        Args:
            element: The element to convert.

        Returns:
            A list containing a single Document object.

        """
        return Document(
            page_content=(parse_output(element, self.output_type)),
            metadata={
                "page": element["page"],
                "id": element["id"],
                "type": self.output_type,
                "split": self.split,
            },
        )

    def _page_document(self, elements) -> List[Document]:
        """
        Combines elements with the same page number into a single Document object.

        Args:
            elements (List[Dict]): A list of elements containing page numbers.

        Returns:
            List[Document]: A list of Document objects, each representing a page
                            with its content and metadata.
        """
        _docs = []
        pages = sorted(set(map(lambda x: x["page"], elements)))

        page_group = [
            [element for element in elements if element["page"] == x] for x in pages
        ]

        for group in page_group:
            page_content = " ".join(
                [parse_output(element, self.output_type) for element in group]
            )

            _docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "page": group[0]["page"],
                        "type": self.output_type,
                        "split": self.split,
                    },
                )
            )

        return _docs

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads the document content based on the specified split type.

        Returns:
            Generator[Document]: A generator that yields Document objects containing
                                 the loaded content.
        """
        full_docs = fitz.open(self.file_path)
        number_of_pages = full_docs.page_count

        if self.split == "none":
            if full_docs.is_pdf:
                result = ""
                start_page = 0
                split_pages = LIMIT_OF_PAGE_REQUEST
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    result += parse_output(response, self.output_type)

                    start_page += split_pages

            else:
                files = {"document": open(self.file_path, "rb")}
                response = self._get_response(files)
                result = parse_output(response, self.output_type)

            yield Document(
                page_content=result,
                metadata={
                    "total_pages": response["billed_pages"],
                    "type": self.output_type,
                    "split": self.split,
                },
            )

        elif self.split == "element":
            if full_docs.is_pdf:
                start_page = 0
                split_pages = 1
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    for element in response["elements"]:
                        yield self._element_document(element)

                    start_page += split_pages

            else:
                files = {"document": open(self.file_path, "rb")}
                response = self._get_response(files)

                docs = []
                for element in response["elements"]:
                    docs.append(self._element_document(element))

                yield from docs

        elif self.split == "page":
            if full_docs.is_pdf:
                start_page = 0
                split_pages = 1
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    elements = response["elements"]
                    yield from self._page_document(elements)

                    start_page += split_pages
            else:
                files = {"document": open(self.file_path, "rb")}
                response = self._get_response(files)
                elements = response["elements"]

                yield from self._page_document(elements)

        else:
            raise ValueError(f"Invalid split type: {self.split}")

    def load(self) -> List[Document]:
        """
        Loads the document and returns a list of Document objects based on
        the specified split type.

        Returns:
            A list of Document objects.

        Raises:
            ValueError: If the split type is invalid.
        """
        full_docs = fitz.open(self.file_path)
        number_of_pages = full_docs.page_count

        if self.split == "none":
            if full_docs.is_pdf:
                result = ""
                start_page = 0
                split_pages = LIMIT_OF_PAGE_REQUEST
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    result += parse_output(response, self.output_type)

                    start_page += split_pages

            else:
                files = {"document": open(self.file_path, "rb")}
                response = self._get_response(files)
                result = parse_output(response, self.output_type)

            return [
                Document(
                    page_content=result,
                    metadata={
                        "total_pages": response["billed_pages"],
                        "type": self.output_type,
                        "split": self.split,
                    },
                )
            ]

        elif self.split == "element":
            docs = []
            if full_docs.is_pdf:
                start_page = 0
                split_pages = LIMIT_OF_PAGE_REQUEST
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    for element in response["elements"]:
                        docs.append(self._element_document(element))

                    start_page += split_pages

            else:
                files = {"document": open(self.file_path, "rb")}
                response = self._get_response(files)

                for element in response["elements"]:
                    docs.append(self._element_document(element))

            return docs

        elif self.split == "page":
            docs = []
            if full_docs.is_pdf:
                start_page = 0
                split_pages = LIMIT_OF_PAGE_REQUEST
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    response = self._split_and_request(
                        full_docs, start_page, split_pages
                    )
                    elements = response["elements"]
                    docs.extend(self._page_document(elements))

                    start_page += split_pages
            else:
                files = {"document": open(self.file_path, "rb")}
                response = self._get_response(files)
                elements = response["elements"]
                docs.extend(self._page_document(elements))

            return docs

        else:
            raise ValueError(f"Invalid split type: {self.split}")
