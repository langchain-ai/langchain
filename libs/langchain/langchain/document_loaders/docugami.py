import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.pydantic_v1 import BaseModel, root_validator

TABLE_NAME = "{http://www.w3.org/1999/xhtml}table"

XPATH_KEY = "xpath"
DOCUMENT_ID_KEY = "id"
DOCUMENT_SOURCE_KEY = "source"
DOCUMENT_NAME_KEY = "name"
STRUCTURE_KEY = "structure"
TAG_KEY = "tag"
PROJECTS_KEY = "projects"

DEFAULT_API_ENDPOINT = "https://api.docugami.com/v1preview1"
DEFAULT_MAX_TEXT_LENGTH = 1024
DEFAULT_MIN_TEXT_LENGTH = 32

logger = logging.getLogger(__name__)


class DocugamiLoader(BaseLoader, BaseModel):
    """Load from `Docugami`.

    To use, you should have the ``dgml-utils`` python package installed.
    """

    api: str = DEFAULT_API_ENDPOINT
    """The Docugami API endpoint to use."""

    access_token: Optional[str] = os.environ.get("DOCUGAMI_API_KEY")
    """The Docugami API access token to use."""

    max_text_length = DEFAULT_MAX_TEXT_LENGTH
    """Max length of chunk and metadata values."""

    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH
    """Threshold under which chunks are appended to next chunk to avoid over-chunking."""

    xml_mode: bool = False
    """Set to true for XML tags in chunk output text."""

    parent_hierarchy_levels: int = 0
    """Set appropriately to get parent chunks using the chunk hierarchy."""

    sub_chunk_tables: bool = False
    """Set to True to return sub-chunks within tables."""

    whitespace_normalize_text: bool = True
    """Set to False if you want to full whitespace formatting in the original XML doc, including indentation."""

    docset_id: Optional[str]
    """The Docugami API docset ID to use."""

    document_ids: Optional[Sequence[str]]
    """The Docugami API document IDs to use."""

    file_paths: Optional[Sequence[Union[Path, str]]]
    """The local file paths to use."""

    fetch_metadata: bool = True
    """Set to False if you don't want to fetch project metadata."""

    @root_validator
    def validate_local_or_remote(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either local file paths are given, or remote API docset ID.

        Args:
            values: The values to validate.

        Returns:
            The validated values.
        """
        if values.get("file_paths") and values.get("docset_id"):
            raise ValueError("Cannot specify both file_paths and remote API docset_id")

        if not values.get("file_paths") and not values.get("docset_id"):
            raise ValueError("Must specify either file_paths or remote API docset_id")

        if values.get("docset_id") and not values.get("access_token"):
            raise ValueError("Must specify access token if using remote API docset_id")

        return values

    def _parse_dgml(
        self, document: Mapping, content: bytes, doc_metadata: Optional[Mapping] = None
    ) -> List[Document]:
        """Parse a single DGML document into a list of Documents."""
        try:
            from lxml import etree
        except ImportError:
            raise ValueError(
                "Could not import lxml python package. "
                "Please install it with `pip install lxml`."
            )

        try:
            from dgml_utils.models import Chunk
            from dgml_utils.segmentation import get_chunks
        except ImportError:
            raise ValueError(
                "Could not import from dgml-utils python package. "
                "Please install it with `pip install dgml-utils`."
            )

        def _build_framework_chunk(dg_chunk: Chunk) -> Document:
            metadata = {
                XPATH_KEY: dg_chunk.xpath,
                DOCUMENT_ID_KEY: document[DOCUMENT_ID_KEY],
                DOCUMENT_NAME_KEY: document[DOCUMENT_NAME_KEY],
                DOCUMENT_SOURCE_KEY: document[DOCUMENT_NAME_KEY],
                STRUCTURE_KEY: dg_chunk.structure,
                TAG_KEY: dg_chunk.tag,
            }

            if doc_metadata:
                metadata.update(doc_metadata)

            return Document(
                page_content=dg_chunk.text,
                metadata=metadata,
            )

        # Parse the tree and return chunks
        tree = etree.parse(io.BytesIO(content))
        root = tree.getroot()

        framework_chunks: List[Document] = []
        dg_chunks = get_chunks(
            root,
            min_text_length=self.min_text_length,
            max_text_length=self.max_text_length,
            whitespace_normalize_text=self.whitespace_normalize_text,
            sub_chunk_tables=self.sub_chunk_tables,
            xml_mode=self.xml_mode,
            parent_hierarchy_levels=self.parent_hierarchy_levels
        )

        for dg_chunk in dg_chunks:
            framework_chunk = _build_framework_chunk(dg_chunk)
            if hasattr(dg_chunk, "parent") and dg_chunk.parent:
                framework_chunk.parent = _build_framework_chunk(dg_chunk.parent)
            framework_chunks.append(framework_chunk)

        return framework_chunks

    def _document_details_for_docset_id(self, docset_id: str) -> List[Dict]:
        """Gets all document details for the given docset ID"""
        url = f"{self.api}/docsets/{docset_id}/documents"
        all_documents = []

        while url:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.access_token}"},
            )
            if response.ok:
                data = response.json()
                all_documents.extend(data["documents"])
                url = data.get("next", None)
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        return all_documents

    def _project_details_for_docset_id(self, docset_id: str) -> List[Dict]:
        """Gets all project details for the given docset ID"""
        url = f"{self.api}/projects?docset.id={docset_id}"
        all_projects = []

        while url:
            response = requests.request(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                data={},
            )
            if response.ok:
                data = response.json()
                all_projects.extend(data["projects"])
                url = data.get("next", None)
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        return all_projects

    def _metadata_for_project(self, project: Dict) -> Dict:
        """Gets project metadata for all files"""
        project_id = project.get("id")

        url = f"{self.api}/projects/{project_id}/artifacts/latest"
        all_artifacts = []

        per_file_metadata = {}
        while url:
            response = requests.request(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.access_token}"},
                data={},
            )
            if response.ok:
                data = response.json()
                all_artifacts.extend(data["artifacts"])
                url = data.get("next", None)
            elif response.status_code == 404:
                # Not found is ok, just means no published projects
                return per_file_metadata
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        for artifact in all_artifacts:
            artifact_name = artifact.get("name")
            artifact_url = artifact.get("url")
            artifact_doc = artifact.get("document")

            if artifact_name == "report-values.xml" and artifact_url and artifact_doc:
                doc_id = artifact_doc["id"]
                metadata: Dict = {}

                # The evaluated XML for each document is named after the project
                response = requests.request(
                    "GET",
                    f"{artifact_url}/content",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    data={},
                )

                if response.ok:
                    try:
                        from lxml import etree
                    except ImportError:
                        raise ImportError(
                            "Could not import lxml python package. "
                            "Please install it with `pip install lxml`."
                        )
                    artifact_tree = etree.parse(io.BytesIO(response.content))
                    artifact_root = artifact_tree.getroot()
                    ns = artifact_root.nsmap
                    entries = artifact_root.xpath("//pr:Entry", namespaces=ns)
                    for entry in entries:
                        heading = entry.xpath("./pr:Heading", namespaces=ns)[0].text
                        value = " ".join(
                            entry.xpath("./pr:Value", namespaces=ns)[0].itertext()
                        ).strip()
                        metadata[heading] = value[: self.max_text_length]
                    per_file_metadata[doc_id] = metadata
                else:
                    raise Exception(
                        f"Failed to download {artifact_url}/content "
                        + "(status: {response.status_code})"
                    )

        return per_file_metadata

    def _load_chunks_for_document(
        self, docset_id: str, document: Dict, doc_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """Load chunks for a document."""
        document_id = document["id"]
        url = f"{self.api}/docsets/{docset_id}/documents/{document_id}/dgml"

        response = requests.request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )

        if response.ok:
            return self._parse_dgml(document, response.content, doc_metadata)
        else:
            raise Exception(
                f"Failed to download {url} (status: {response.status_code})"
            )

    def load(self) -> List[Document]:
        """Load documents."""
        chunks: List[Document] = []

        if self.access_token and self.docset_id:
            # remote mode
            _document_details = self._document_details_for_docset_id(self.docset_id)
            if self.document_ids:
                _document_details = [
                    d for d in _document_details if d["id"] in self.document_ids
                ]

            _project_details = self._project_details_for_docset_id(self.docset_id)
            combined_project_metadata = {}
            if _project_details and self.fetch_metadata:
                # if there are any projects for this docset, load project metadata
                for project in _project_details:
                    metadata = self._metadata_for_project(project)
                    combined_project_metadata.update(metadata)

            for doc in _document_details:
                doc_metadata = combined_project_metadata.get(doc["id"])
                chunks += self._load_chunks_for_document(
                    self.docset_id, doc, doc_metadata
                )
        elif self.file_paths:
            # Local mode (for integration testing, or pre-downloaded XML)
            for path in self.file_paths:
                path = Path(path)
                with open(path, "rb") as file:
                    chunks += self._parse_dgml(
                        {
                            DOCUMENT_ID_KEY: path.name,
                            DOCUMENT_NAME_KEY: path.name,
                        },
                        file.read(),
                    )

        return chunks
