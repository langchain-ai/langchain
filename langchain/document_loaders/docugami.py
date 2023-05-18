"""Loader that loads processed documents from Docugami."""

import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import requests
from pydantic import BaseModel, root_validator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

TD_NAME = "{http://www.w3.org/1999/xhtml}td"
TABLE_NAME = "{http://www.w3.org/1999/xhtml}table"

XPATH_KEY = "xpath"
DOCUMENT_ID_KEY = "id"
DOCUMENT_NAME_KEY = "name"
STRUCTURE_KEY = "structure"
TAG_KEY = "tag"
PROJECTS_KEY = "projects"

DEFAULT_API_ENDPOINT = "https://api.docugami.com/v1preview1"

logger = logging.getLogger(__name__)


class DocugamiLoader(BaseLoader, BaseModel):
    """Loader that loads processed docs from Docugami.

    To use, you should have the ``lxml`` python package installed.
    """

    api: str = DEFAULT_API_ENDPOINT

    access_token: Optional[str] = os.environ.get("DOCUGAMI_API_KEY")
    docset_id: Optional[str]
    document_ids: Optional[Sequence[str]]
    file_paths: Optional[Sequence[Union[Path, str]]]
    min_chunk_size: int = 32  # appended to the next chunk to avoid over-chunking

    @root_validator
    def validate_local_or_remote(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either local file paths are given, or remote API docset ID."""
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

        # helpers
        def _xpath_qname_for_chunk(chunk: Any) -> str:
            """Get the xpath qname for a chunk."""
            qname = f"{chunk.prefix}:{chunk.tag.split('}')[-1]}"

            parent = chunk.getparent()
            if parent is not None:
                doppelgangers = [x for x in parent if x.tag == chunk.tag]
                if len(doppelgangers) > 1:
                    idx_of_self = doppelgangers.index(chunk)
                    qname = f"{qname}[{idx_of_self + 1}]"

            return qname

        def _xpath_for_chunk(chunk: Any) -> str:
            """Get the xpath for a chunk."""
            ancestor_chain = chunk.xpath("ancestor-or-self::*")
            return "/" + "/".join(_xpath_qname_for_chunk(x) for x in ancestor_chain)

        def _structure_value(node: Any) -> str:
            """Get the structure value for a node."""
            structure = (
                "table"
                if node.tag == TABLE_NAME
                else node.attrib["structure"]
                if "structure" in node.attrib
                else None
            )
            return structure

        def _is_structural(node: Any) -> bool:
            """Check if a node is structural."""
            return _structure_value(node) is not None

        def _is_heading(node: Any) -> bool:
            """Check if a node is a heading."""
            structure = _structure_value(node)
            return structure is not None and structure.lower().startswith("h")

        def _get_text(node: Any) -> str:
            """Get the text of a node."""
            return " ".join(node.itertext()).strip()

        def _has_structural_descendant(node: Any) -> bool:
            """Check if a node has a structural descendant."""
            for child in node:
                if _is_structural(child) or _has_structural_descendant(child):
                    return True
            return False

        def _leaf_structural_nodes(node: Any) -> List:
            """Get the leaf structural nodes of a node."""
            if _is_structural(node) and not _has_structural_descendant(node):
                return [node]
            else:
                leaf_nodes = []
                for child in node:
                    leaf_nodes.extend(_leaf_structural_nodes(child))
                return leaf_nodes

        def _create_doc(node: Any, text: str) -> Document:
            """Create a Document from a node and text."""
            metadata = {
                XPATH_KEY: _xpath_for_chunk(node),
                DOCUMENT_ID_KEY: document["id"],
                DOCUMENT_NAME_KEY: document["name"],
                STRUCTURE_KEY: node.attrib.get("structure", ""),
                TAG_KEY: re.sub(r"\{.*\}", "", node.tag),
            }

            if doc_metadata:
                metadata.update(doc_metadata)

            return Document(
                page_content=text,
                metadata=metadata,
            )

        # parse the tree and return chunks
        tree = etree.parse(io.BytesIO(content))
        root = tree.getroot()

        chunks: List[Document] = []
        prev_small_chunk_text = None
        for node in _leaf_structural_nodes(root):
            text = _get_text(node)
            if prev_small_chunk_text:
                text = prev_small_chunk_text + " " + text
                prev_small_chunk_text = None

            if _is_heading(node) or len(text) < self.min_chunk_size:
                # Save headings or other small chunks to be appended to the next chunk
                prev_small_chunk_text = text
            else:
                chunks.append(_create_doc(node, text))

        if prev_small_chunk_text and len(chunks) > 0:
            # small chunk at the end left over, just append to last chunk
            chunks[-1].page_content += " " + prev_small_chunk_text

        return chunks

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
            else:
                raise Exception(
                    f"Failed to download {url} (status: {response.status_code})"
                )

        per_file_metadata = {}
        for artifact in all_artifacts:
            artifact_name = artifact.get("name")
            artifact_url = artifact.get("url")
            artifact_doc = artifact.get("document")

            if artifact_name == f"{project_id}.xml" and artifact_url and artifact_doc:
                doc_id = artifact_doc["id"]
                metadata: Dict = {}

                # the evaluated XML for each document is named after the project
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
                        raise ValueError(
                            "Could not import lxml python package. "
                            "Please install it with `pip install lxml`."
                        )
                    artifact_tree = etree.parse(io.BytesIO(response.content))
                    artifact_root = artifact_tree.getroot()
                    ns = artifact_root.nsmap
                    entries = artifact_root.xpath("//wp:Entry", namespaces=ns)
                    for entry in entries:
                        heading = entry.xpath("./wp:Heading", namespaces=ns)[0].text
                        value = " ".join(
                            entry.xpath("./wp:Value", namespaces=ns)[0].itertext()
                        ).strip()
                        metadata[heading] = value
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
            if _project_details:
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
            # local mode (for integration testing, or pre-downloaded XML)
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
