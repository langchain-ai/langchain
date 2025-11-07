"""LangSmith document loader."""

import datetime
import json
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langsmith import Client as LangSmithClient
from typing_extensions import override

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class LangSmithLoader(BaseLoader):
    """Load LangSmith Dataset examples as `Document` objects.

    Loads the example inputs as the `Document` page content and places the entire
    example into the `Document` metadata. This allows you to easily create few-shot
    example retrievers from the loaded documents.

    ??? note "Lazy loading example"

        ```python
        from langchain_core.document_loaders import LangSmithLoader

        loader = LangSmithLoader(dataset_id="...", limit=100)
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        ```

        ```python
        # -> [Document("...", metadata={"inputs": {...}, "outputs": {...}, ...}), ...]
        ```
    """

    def __init__(
        self,
        *,
        dataset_id: uuid.UUID | str | None = None,
        dataset_name: str | None = None,
        example_ids: Sequence[uuid.UUID | str] | None = None,
        as_of: datetime.datetime | str | None = None,
        splits: Sequence[str] | None = None,
        inline_s3_urls: bool = True,
        offset: int = 0,
        limit: int | None = None,
        metadata: dict | None = None,
        filter: str | None = None,  # noqa: A002
        content_key: str = "",
        format_content: Callable[..., str] | None = None,
        client: LangSmithClient | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create a LangSmith loader.

        Args:
            dataset_id: The ID of the dataset to filter by.
            dataset_name: The name of the dataset to filter by.
            content_key: The inputs key to set as Document page content. `'.'` characters
                are interpreted as nested keys. E.g. `content_key="first.second"` will
                result in
                `Document(page_content=format_content(example.inputs["first"]["second"]))`
            format_content: Function for converting the content extracted from the example
                inputs into a string. Defaults to JSON-encoding the contents.
            example_ids: The IDs of the examples to filter by.
            as_of: The dataset version tag or timestamp to retrieve the examples as of.
                Response examples will only be those that were present at the time of
                the tagged (or timestamped) version.
            splits: A list of dataset splits, which are
                divisions of your dataset such as `train`, `test`, or `validation`.
                Returns examples only from the specified splits.
            inline_s3_urls: Whether to inline S3 URLs.
            offset: The offset to start from.
            limit: The maximum number of examples to return.
            metadata: Metadata to filter by.
            filter: A structured filter string to apply to the examples.
            client: LangSmith Client. If not provided will be initialized from below args.
            client_kwargs: Keyword args to pass to LangSmith client init. Should only be
                specified if `client` isn't.

        Raises:
            ValueError: If both `client` and `client_kwargs` are provided.
        """  # noqa: E501
        if client and client_kwargs:
            raise ValueError
        self._client = client or LangSmithClient(**client_kwargs)
        self.content_key = list(content_key.split(".")) if content_key else []
        self.format_content = format_content or _stringify
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.example_ids = example_ids
        self.as_of = as_of
        self.splits = splits
        self.inline_s3_urls = inline_s3_urls
        self.offset = offset
        self.limit = limit
        self.metadata = metadata
        self.filter = filter

    @override
    def lazy_load(self) -> Iterator[Document]:
        for example in self._client.list_examples(
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            example_ids=self.example_ids,
            as_of=self.as_of,
            splits=self.splits,
            inline_s3_urls=self.inline_s3_urls,
            offset=self.offset,
            limit=self.limit,
            metadata=self.metadata,
            filter=self.filter,
        ):
            content: Any = example.inputs
            for key in self.content_key:
                content = content[key]
            content_str = self.format_content(content)
            metadata = example.dict()
            # Stringify datetime and UUID types.
            for k in ("dataset_id", "created_at", "modified_at", "source_run_id", "id"):
                metadata[k] = str(metadata[k]) if metadata[k] else metadata[k]
            yield Document(content_str, metadata=metadata)


def _stringify(x: str | dict) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, indent=2)
    except Exception:
        return str(x)
