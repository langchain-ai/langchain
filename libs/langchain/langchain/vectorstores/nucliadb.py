import os
from typing import Any, Dict, Iterable, List, Optional, Type

from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from langchain.vectorstores.base import VST, VectorStore

FIELD_TYPES = {
    "f": "files",
    "t": "texts",
    "l": "links",
}


class NucliaDB(VectorStore):
    """NucliaDB vector store."""

    _config: Dict[str, Any] = {}

    def __init__(
        self,
        knowledge_box: str,
        local: bool,
        api_key: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None:
        """Initialize the NucliaDB client.

        Args:
            knowledge_box: the Knowledge Box id.
            local: Whether to use a local NucliaDB instance or Nuclia Cloud
            api_key: A contributor API key for the kb (needed when local is False)
            backend: The backend url to use when local is True, defaults to
            http://localhost:8080
        """
        try:
            from nuclia.sdk import NucliaAuth
        except ImportError:
            raise ValueError(
                "nuclia python package not found. "
                "Please install it with `pip install nuclia`."
            )
        self._config["LOCAL"] = local
        zone = os.environ.get("NUCLIA_ZONE", "europe-1")
        self._kb = knowledge_box
        if local:
            if not backend:
                backend = "http://localhost:8080"
            self._config["BACKEND"] = f"{backend}/api/v1"
            self._config["TOKEN"] = None
            NucliaAuth().nucliadb(url=backend)
            NucliaAuth().kb(url=self.kb_url, interactive=False)
        else:
            self._config["BACKEND"] = f"https://{zone}.nuclia.cloud/api/v1"
            self._config["TOKEN"] = api_key
            NucliaAuth().kb(
                url=self.kb_url, token=self._config["TOKEN"], interactive=False
            )

    @property
    def is_local(self) -> str:
        return self._config["LOCAL"]

    @property
    def kb_url(self) -> str:
        return f"{self._config['BACKEND']}/kb/{self._kb}"

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload texts to NucliaDB"""
        ids = []
        from nuclia.sdk import NucliaResource

        factory = NucliaResource()
        for i, text in enumerate(texts):
            extra: Dict[str, Any] = {"metadata": ""}
            if metadatas:
                extra = {"metadata": metadatas[i]}
            id = factory.create(
                texts={"text": {"body": text}},
                extra=extra,
                url=self.kb_url,
                api_key=self._config["TOKEN"],
            )
            ids.append(id)
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return None
        from nuclia.sdk import NucliaResource

        factory = NucliaResource()
        results: List[bool] = []
        for id in ids:
            try:
                factory.delete(rid=id, url=self.kb_url, api_key=self._config["TOKEN"])
                results.append(True)
            except ValueError:
                results.append(False)
        return all(results)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        from nuclia.sdk import NucliaSearch
        from nucliadb_models.search import FindRequest, ResourceProperties

        request = FindRequest(
            query=query,
            page_size=k,
            show=[ResourceProperties.VALUES, ResourceProperties.EXTRA],
        )
        search = NucliaSearch()
        results = search.find(
            query=request, url=self.kb_url, api_key=self._config["TOKEN"]
        )
        paragraphs = []
        for resource in results.resources.values():
            for field in resource.fields.values():
                for paragraph_id, paragraph in field.paragraphs.items():
                    info = paragraph_id.split("/")
                    field_type = FIELD_TYPES.get(info[1], None)
                    field_id = info[2]
                    if not field_type:
                        continue
                    value = getattr(resource.data, field_type, {}).get(field_id, None)
                    paragraphs.append(
                        {
                            "text": paragraph.text,
                            "metadata": {
                                "extra": getattr(
                                    getattr(resource, "extra", {}), "metadata", None
                                ),
                                "value": value,
                            },
                            "order": paragraph.order,
                        }
                    )
        sorted_paragraphs = sorted(paragraphs, key=lambda x: x["order"])
        return [
            Document(page_content=paragraph["text"], metadata=paragraph["metadata"])
            for paragraph in sorted_paragraphs
        ]

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings."""
        raise NotImplementedError
