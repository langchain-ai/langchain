import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class JSONLoader(BaseLoader):
    """
    Load a `JSON` file using a `jq` schema.

    Setup:
        .. code-block:: bash

            pip install -U jq

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import JSONLoader
            import json
            from pathlib import Path

            file_path='./sample_quiz.json'
            data = json.loads(Path(file_path).read_text())
            loader = JSONLoader(
                     file_path=file_path,
                     jq_schema='.quiz',
                     text_content=False)

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            {"sport": {"q1": {"question": "Which one is correct team name in
            NBA?", "options": ["New York Bulls"
            {'source': '/sample_quiz
            .json', 'seq_num': 1}

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            {"sport": {"q1": {"question": "Which one is correct team name in
            NBA?", "options": ["New York Bulls"
            {'source': '/sample_quizg
            .json', 'seq_num': 1}

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            {"sport": {"q1": {"question": "Which one is correct team name in
            NBA?", "options": ["New York Bulls"
            {'source': '/sample_quiz
            .json', 'seq_num': 1}
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        jq_schema: str,
        content_key: Optional[str] = None,
        is_content_key_jq_parsable: Optional[bool] = False,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
        json_lines: bool = False,
    ):
        """Initialize the JSONLoader.

        Args:
            file_path (Union[str, Path]): The path to the JSON or JSON Lines file.
            jq_schema (str): The jq schema to use to extract the data or text from
                the JSON.
            content_key (str): The key to use to extract the content from
                the JSON if the jq_schema results to a list of objects (dict).
                If is_content_key_jq_parsable is True, this has to be a jq compatible
                schema. If is_content_key_jq_parsable is False, this should be a simple
                string key.
            is_content_key_jq_parsable (bool): A flag to determine if
                content_key is parsable by jq or not. If True, content_key is
                treated as a jq schema and compiled accordingly. If False or if
                content_key is None, content_key is used as a simple string.
                Default is False.
            metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
                object extracted by the jq_schema and the default metadata and returns
                a dict of the updated metadata.
            text_content (bool): Boolean flag to indicate whether the content is in
                string format, default to True.
            json_lines (bool): Boolean flag to indicate whether the input is in
                JSON Lines format.
        """
        try:
            import jq

            self.jq = jq
        except ImportError:
            raise ImportError(
                "jq package not found, please install it with `pip install jq`"
            )

        self.file_path = Path(file_path).resolve()
        self._jq_schema = jq.compile(jq_schema)
        self._is_content_key_jq_parsable = is_content_key_jq_parsable
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content
        self._json_lines = json_lines

    def lazy_load(self) -> Iterator[Document]:
        """Load and return documents from the JSON file."""
        index = 0
        if self._json_lines:
            with self.file_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        for doc in self._parse(line, index):
                            yield doc
                            index += 1
        else:
            for doc in self._parse(self.file_path.read_text(encoding="utf-8"), index):
                yield doc
                index += 1

    def _parse(self, content: str, index: int) -> Iterator[Document]:
        """Convert given content to documents."""
        data = self._jq_schema.input(json.loads(content))

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.
        if self._content_key is not None:
            self._validate_content_key(data)
        # if self._metadata_func is not None:
        #     self._validate_metadata_func(data)

        for i, sample in enumerate(data, index + 1):
            text = self._get_text(sample=sample)
            metadata = self._get_metadata(
                sample=sample, source=str(self.file_path), seq_num=i
            )
            yield Document(page_content=text, metadata=metadata)

    def _get_text(self, sample: Any) -> str:
        """Convert sample to string format"""
        if self._content_key is not None:
            if self._is_content_key_jq_parsable:
                compiled_content_key = self.jq.compile(self._content_key)
                content = compiled_content_key.input(sample).first()
            else:
                content = sample[self._content_key]
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

    def _get_metadata(
        self, sample: Dict[str, Any], **additional_fields: Any
    ) -> Dict[str, Any]:
        """
        Return a metadata dictionary base on the existence of metadata_func
        :param sample: single data payload
        :param additional_fields: key-word arguments to be added as metadata values
        :return:
        """
        # if self._metadata_func is not None:
        #     return self._metadata_func(sample, additional_fields)
        # else:
        #     return additional_fields

        if self._metadata_func is not None:
            result = self._metadata_func(sample, additional_fields)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Expected the metadata_func to return a dict but got \
                                `{type(result)}`"
                )
            return result
        else:
            return additional_fields

    def _validate_content_key(self, data: Any) -> None:
        """Check if a content key is valid"""

        sample = data.first()
        if not isinstance(sample, dict):
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict), \
                    so sample must be a dict but got `{type(sample)}`"
            )

        if (
            not self._is_content_key_jq_parsable
            and sample.get(self._content_key) is None
        ):
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict) \
                    with the key `{self._content_key}`"
            )
        if (
            self._is_content_key_jq_parsable
            and self.jq.compile(self._content_key).input(sample).text() is None
        ):
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict) \
                    with the key `{self._content_key}` which should be parsable by jq"
            )

    # def _validate_metadata_func(self, data: Any) -> None:
    #     """Check if the metadata_func output is valid"""
    #
    #     sample = data.first()
    #     if self._metadata_func is not None:
    #         sample_metadata = self._metadata_func(sample,
    #         {"source": ".", "seq_num": 0})
    #         if not isinstance(sample_metadata, dict):
    #             raise ValueError(
    #                 f"Expected the metadata_func to return a dict but got \
    #                     `{type(sample_metadata)}`"
    #             )
