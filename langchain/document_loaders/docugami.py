"""Loader that loads processed documents from Docugami."""

import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, root_validator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

TD_NAME = "{http://www.w3.org/1999/xhtml}td"
TABLE_NAME = "{http://www.w3.org/1999/xhtml}table"

XPATH_KEY = "xpath"
DOCUMENT_ID_KEY = "documentId"
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
    document_ids: Optional[List[str]]
    file_paths: Optional[List[Path]]

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

    def _xpath_for_chunk(self, chunk: Any) -> str:
        ancestor_chain = chunk.xpath("ancestor-or-self::*")

        def _xpath_qname_for_chunk(chunk: Any) -> str:
            qname = f"{chunk.prefix}:{chunk.tag.split('}')[-1]}"

            parent = chunk.getparent()
            if parent is not None:
                doppelgangers = [x for x in parent if x.tag == chunk.tag]
                if len(doppelgangers) > 1:
                    idx_of_self = doppelgangers.index(chunk)
                    qname = f"{qname}[{idx_of_self + 1}]"

            return qname

        return "/" + "/".join(_xpath_qname_for_chunk(x) for x in ancestor_chain)

    def _parse_dgml(
        self, document_id: str, content: bytes, project_metadata: Optional[List] = None
    ) -> List[Document]:
        try:
            from lxml import etree
        except ImportError:
            raise ValueError(
                "Could not import lxml python package. "
                "Please install it with `pip install lxml`."
            )

        # combine metadata across projects for this document
        doc_metadata = []
        if project_metadata:
            for metadata in project_metadata:
                project_doc_metadata = metadata.get(document_id)
                if project_doc_metadata:
                    doc_metadata.append(project_doc_metadata)

        # parse the tree and return chunks
        tree = etree.parse(io.BytesIO(content))
        root = tree.getroot()
        chunks: List[Document] = []
        for chunk in root.iter(tag=etree.Element):
            structure = chunk.attrib.get("structure", "")
            valid_chunk = False
            if structure in ("h1", "p"):
                # heading or paragraph chunks found in document layout
                valid_chunk = True
            elif chunk.tag == TD_NAME or structure in ("div", "li"):
                if not any(
                    "structure" in x.attrib or x.tag == TABLE_NAME for x in chunk
                ):
                    # Table cells, divs and list items found in document
                    # layout that don't contain further inner structure
                    valid_chunk = True
                    if chunk.tag == TD_NAME:
                        structure = "td"

            if valid_chunk:
                chunks.append(
                    Document(
                        page_content=" ".join(chunk.itertext()).strip(),
                        metadata={
                            XPATH_KEY: self._xpath_for_chunk(chunk),
                            DOCUMENT_ID_KEY: document_id,
                            STRUCTURE_KEY: structure,
                            TAG_KEY: re.sub(r"\{.*\}", "", chunk.tag),
                            PROJECTS_KEY: doc_metadata,
                        },
                    )
                )

        return chunks

    def _document_ids_for_docset_id(self, docset_id: str) -> List[str]:
        """Gets all document IDs for the given docset ID"""
        url = f"{self.api}/docsets/{docset_id}/documents"
        response = requests.request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )
        if response.ok:
            # TODO: pagination
            return [doc["id"] for doc in response.json()["documents"]]
        else:
            raise Exception(
                f"Failed to download {url} (status: {response.status_code})"
            )

    def _project_details_for_docset_id(self, docset_id: str) -> List[Dict]:
        """Gets all project details for the given docset ID"""
        url = f"{self.api}/projects?docset.id={docset_id}"
        response = requests.request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )
        if response.ok:
            # TODO: pagination
            return response.json()["projects"]
        else:
            raise Exception(
                f"Failed to download {url} (status: {response.status_code})"
            )

    def _metadata_for_project(self, project: Dict) -> List[Dict]:
        """Gets project metadata"""
        project_id = project.get("id")
        project_name = project.get("name")
        project_type = project.get("type")

        per_file_metadata = {}
        url = f"{self.api}/projects/{project_id}/artifacts/latest"
        response = requests.request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )
        if response.ok:
            # TODO: pagination
            artifacts = response.json()["artifacts"]
            for artifact in artifacts:
                artifact_name = artifact.get("name")
                artifact_url = artifact.get("url")
                artifact_doc = artifact.get("document")

                if (
                    artifact_name == f"{project_id}.xml"
                    and artifact_url
                    and artifact_doc
                ):
                    doc_id = artifact_doc["id"]
                    metadata: Dict = {
                        "projectType": project_type,
                        "projectTitle": project_name,
                        "entries": [],
                    }

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
                            xpath = entry.xpath("./wp:XPath", namespaces=ns)[0].text
                            value = " ".join(
                                entry.xpath("./wp:Value", namespaces=ns)[0].itertext()
                            ).strip()
                            metadata["entries"].append(  # pylint: disable
                                {
                                    "heading": heading,
                                    "xpath": xpath,
                                    "value": value,
                                }
                            )
                        per_file_metadata[doc_id] = metadata
                    else:
                        raise Exception(
                            f"Failed to download {artifact_url}/content "
                            + "(status: {response.status_code})"
                        )

            return per_file_metadata
        else:
            raise Exception(
                f"Failed to download {url} (status: {response.status_code})"
            )

    def _load_chunks_for_document_id(
        self, docset_id: str, document_id: str, project_metadata: Optional[List] = None
    ) -> List[Document]:
        url = f"{self.api}/docsets/{docset_id}/documents/{document_id}/dgml"

        response = requests.request(
            "GET",
            url,
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )

        if response.ok:
            return self._parse_dgml(document_id, response.content, project_metadata)
        else:
            raise Exception(
                f"Failed to download {url} (status: {response.status_code})"
            )

    def load(self) -> List[Document]:
        """Load documents."""
        chunks: List[Document] = []

        if self.access_token and self.docset_id:
            # remote mode
            _document_ids = self.document_ids
            if not _document_ids:
                # no document IDs specified, default to all docs in docset
                _document_ids = self._document_ids_for_docset_id(self.docset_id)

            _project_details = self._project_details_for_docset_id(self.docset_id)
            project_metadatas = []
            if _project_details:
                # if there are any projects for this docset, load project metadata
                for project in _project_details:
                    metadata = self._metadata_for_project(project)
                    project_metadatas.append(metadata)

            for doc_id in _document_ids:
                chunks += self._load_chunks_for_document_id(
                    self.docset_id, doc_id, project_metadatas
                )
        elif self.file_paths:
            # local mode (for integration testing, or pre-downloaded XML)
            for path in self.file_paths:
                with open(path, "rb") as file:
                    chunks += self._parse_dgml(path.name, file.read())

        return chunks
