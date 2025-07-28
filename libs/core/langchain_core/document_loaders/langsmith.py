"""LangSmith document loader."""

import datetime
import json
import uuid
from collections.abc import Iterator, Sequence
from typing import Any, Callable, Optional, Union

from langsmith import Client as LangSmithClient
from typing_extensions import override

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class LangSmithLoader(BaseLoader):
    """Load LangSmith Dataset examples as Documents.

    Loads the example inputs as the Document page content and places the entire example
    into the Document metadata. This allows you to easily create few-shot example
    retrievers from the loaded documents.

    .. dropdown:: Lazy load

        .. code-block:: python

            from langchain_core.document_loaders import LangSmithLoader

            loader = LangSmithLoader(dataset_id="...", limit=100)
            docs = []
            for doc in loader.lazy_load():
                docs.append(doc)

        .. code-block:: pycon

            # -> [Document("...", metadata={"inputs": {...}, "outputs": {...}, ...}), ...]

    .. versionadded:: 0.2.34

    """  # noqa: E501

    def __init__(
        self,
        *,
        dataset_id: Optional[Union[uuid.UUID, str]] = None,
        dataset_name: Optional[str] = None,
        example_ids: Optional[Sequence[Union[uuid.UUID, str]]] = None,
        as_of: Optional[Union[datetime.datetime, str]] = None,
        splits: Optional[Sequence[str]] = None,
        inline_s3_urls: bool = True,
        offset: int = 0,
        limit: Optional[int] = None,
        metadata: Optional[dict] = None,
        filter: Optional[str] = None,  # noqa: A002
        content_key: str = "",
        format_content: Optional[Callable[..., str]] = None,
        client: Optional[LangSmithClient] = None,
        **client_kwargs: Any,
    ) -> None:
        """Create a LangSmith loader.

        Args:
            dataset_id: The ID of the dataset to filter by. Defaults to None.
            dataset_name: The name of the dataset to filter by. Defaults to None.
            content_key: The inputs key to set as Document page content. ``'.'`` characters
                are interpreted as nested keys. E.g. ``content_key="first.second"`` will
                result in
                ``Document(page_content=format_content(example.inputs["first"]["second"]))``
            format_content: Function for converting the content extracted from the example
                inputs into a string. Defaults to JSON-encoding the contents.
            example_ids: The IDs of the examples to filter by. Defaults to None.
            as_of: The dataset version tag OR
                timestamp to retrieve the examples as of.
                Response examples will only be those that were present at the time
                of the tagged (or timestamped) version.
            splits: A list of dataset splits, which are
                divisions of your dataset such as 'train', 'test', or 'validation'.
                Returns examples only from the specified splits.
            inline_s3_urls: Whether to inline S3 URLs. Defaults to True.
            offset: The offset to start from. Defaults to 0.
            limit: The maximum number of examples to return.
            metadata: Metadata to filter by. Defaults to None.
            filter: A structured filter string to apply to the examples.
            client: LangSmith Client. If not provided will be initialized from below args.
            client_kwargs: Keyword args to pass to LangSmith client init. Should only be
                specified if ``client`` isn't.
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


def _stringify(x: Union[str, dict]) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, indent=2)
    except Exception:
        return str(x)
