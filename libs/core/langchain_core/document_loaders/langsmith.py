import json
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import requests
from langsmith import Client as LangSmithClient
from urllib3 import Retry

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class LangSmithLoader(BaseLoader):
    """Load LangSmith Dataset examples as Documents.

    Example:
        .. code-block:: python

            from langchain_core.document_loaders import LangSmithLoader

            loader = LangSmithLoader(dataset_id="...", limit=100)
            docs = []
            for doc in loader.lazy_load():
                docs.append(doc)

    """

    def __init__(
        self,
        *,
        client: Optional[LangSmithClient] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        retry_config: Optional[Retry] = None,
        timeout_ms: Optional[Union[int, Tuple[int, int]]] = None,
        session: Optional[requests.Session] = None,
        content_key: Optional[str] = None,
        format_content: Optional[Callable[..., str]] = None,
        **list_examples_params: Any,
    ) -> None:
        """Init loader.

        Args:
            ...
        """
        if not client:
            client = LangSmithClient(
                api_url,
                api_key=api_key,
                retry_config=retry_config,
                timeout_ms=timeout_ms,
                session=session,
            )
        self._client = client
        self._list_examples_params = list_examples_params
        self._content_key = list(content_key.split(".")) if content_key else []
        self._format_content = format_content or _stringify

    def lazy_load(self) -> Iterator[Document]:
        for example in self._client.list_examples(**self._list_examples_params):
            content: Any = example.inputs
            for key in self._content_key:
                content = content[key]
            content_str = self._format_content(content)
            yield Document(content_str, metadata=example.dict())


def _stringify(x: Union[str, dict]) -> str:
    if isinstance(x, str):
        return x
    else:
        try:
            return json.dumps(x, indent=2)
        except Exception:
            return str(x)
