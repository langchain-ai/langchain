"""Loader that loads data from JSON."""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class JSONLoader(BaseLoader):
    """Loads a JSON file and references a jq schema provided to load the text into
    documents.

    Example:
        [{"text": ...}, {"text": ...}, {"text": ...}] -> schema = .[].text
        {"key": [{"text": ...}, {"text": ...}, {"text": ...}]} -> schema = .key[].text
        ["", "", ""] -> schema = .[]
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        jq_schema: str,
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
    ):
        """Initialize the JSONLoader.

        Args:
            file_path (Union[str, Path]): The path to the JSON file.
            jq_schema (str): The jq schema to use to extract the data or text from
                the JSON.
            content_key (str): The key to use to extract the content from the JSON if
                the jq_schema results to a list of objects (dict).
            metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
                object extracted by the jq_schema and the default metadata and returns
                a dict of the updated metadata.
            text_content (bool): Boolean flag to indicates whether the content is in
                string format, default to True
        """
        try:
            import jq  # noqa:F401
        except ImportError:
            raise ImportError(
                "jq package not found, please install it with `pip install jq`"
            )

        self.file_path = Path(file_path).resolve()
        self._jq_schema = jq.compile(jq_schema)
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        data = self._jq_schema.input(json.loads(self.file_path.read_text()))

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.
        if self._content_key is not None:
            self._validate_content_key(data)

        docs = []
        for i, sample in enumerate(data, 1):
            metadata = dict(
                source=str(self.file_path),
                seq_num=i,
            )
            text = self._get_text(sample=sample, metadata=metadata)
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def _get_text(self, sample: Any, metadata: dict) -> str:
        """Convert sample to string format"""
        if self._content_key is not None:
            content = sample.get(self._content_key)
            if self._metadata_func is not None:
                # We pass in the metadata dict to the metadata_func
                # so that the user can customize the default metadata
                # based on the content of the JSON object.
                metadata = self._metadata_func(sample, metadata)
        else:
            content = sample

        if self._text_content and not isinstance(content, str):
            raise ValueError(
                f"Expected page_content is string, got {type(content)} instead. \
                    Set `text_content=False` if the desired input for \
                    `page_content` is not a string"
            )

        # In case the text is None, set it to an empty string
        elif isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content) if content else ""
        else:
            return str(content) if content is not None else ""

    def _validate_content_key(self, data: Any) -> None:
        """Check if content key is valid"""
        sample = data.first()
        if not isinstance(sample, dict):
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict), \
                    so sample must be a dict but got `{type(sample)}`"
            )

        if sample.get(self._content_key) is None:
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict) \
                    with the key `{self._content_key}`"
            )

        if self._metadata_func is not None:
            sample_metadata = self._metadata_func(sample, {})
            if not isinstance(sample_metadata, dict):
                raise ValueError(
                    f"Expected the metadata_func to return a dict but got \
                        `{type(sample_metadata)}`"
                )
