import os
from pathlib import Path
from typing import Dict, Iterator, Literal, Optional, Union

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob

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


class UpstageDocumentParser(BaseBlobParser):
    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        output_type: OutputType = "text",
        split: SplitType = "none",
    ):
        """
        Initializes an instance of the Upstage class.

        Args:
            api_key (str, optional): The API key for accessing the Upstage API.
                                     If not provided, it will be retrieved from
                                     the environment variable UPSTAGE_API_KEY.
                                     Defaults to None.
            api_base (str, optional): The base URL for the Upstage API. If not provided,
                                      it will be retrieved from the environment variable
                                      PSTAGE_API_BASE. Defaults to None.
            output_type (OutputType, optional): The type of output to retrieve from the
                                                Upstage API. Defaults to "text".
            split (SplitType, optional): The type of splitting to apply to the output.
                                         Defaults to "none".
        """
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY", api_key, "UPSTAGE_API_KEY"
        )
        self.api_base = get_from_param_or_env(
            "UPSTAGE_API_BASE", api_base, "UPSTAGE_API_BASE", LAYOUT_ANALYZER_URL
        )

        self.output_type = output_type
        self.split = split

        validate_api_key(self.api_key)

    def _get_response(self, file_path: Union[str, Path]) -> Dict:
        """
        Sends a POST request to the API endpoint with the given file path and returns
        the response as a JSON dictionary.

        Args:
            file_path (Union[str, Path]): The path to the file to be sent in
                                          the request.

        Returns:
            Dict: The response from the API as a JSON dictionary.

        Raises:
            ValueError: If there is an error in the API call.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            files = {"document": open(file_path, "rb")}
            response = requests.post(self.api_base, headers=headers, files=files)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API call error: {e}")
        finally:
            files["document"].close()

        return response.json()

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Lazily parses the given blob and yields Document objects based on the split
        type.

        Args:
            blob (Blob): The blob to be parsed.

        Yields:
            Document: A parsed document object.

        Raises:
            ValueError: If the split type is invalid.
        """

        response = self._get_response(blob.path)

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
