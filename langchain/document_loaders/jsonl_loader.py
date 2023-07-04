"""Loader that loads data from JSONL File."""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class JSONLinesLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        jq_schema: str,
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
    ):
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
        """Load and return documents from the JSONL file."""

        json_l_data = self._load_jsonl_file(self.file_path)
        docs = []
        for i, json_item in enumerate(json_l_data, 1):
            data = self._jq_schema.input(json_item)
            # Perform some validation
            # This is not a perfect validation, but it should catch most cases
            # and prevent the user from getting a cryptic error later on.
            if self._content_key is not None:
                self._validate_content_key(data)

            for sample in data:
                metadata = dict(
                    source=str(self.file_path),
                    seq_num=i,
                )
                text = self._get_text(sample=sample, metadata=metadata)
                docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def _load_jsonl_file(self, file_path: Path) -> List[Dict]:
        """Loads and returns json object in list from the given jsonl file"""
        data = []
        with open(file_path, "r") as file:
            for line in file:
                json_data = json.loads(line)
                data.append(json_data)
            file.close()
        return data

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
