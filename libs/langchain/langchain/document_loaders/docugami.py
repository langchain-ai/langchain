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
DEFAULT_MIN_CHUNK_SIZE = 32

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

    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE
    """Threshold under which chunks are appended to next chunk to avoid over-chunking."""

    include_xml_tags: bool = False
    """Set to true for XML tags in chunk output text."""

    xml_hierarchy_levels: int = 0
    """Set appropriately to get parent chunks using the XML hierarchy. Must be 0 if include_xml_tags is False."""

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
            from dgml_utils.segmentation import get_leaf_structural_chunks
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

        # parse the tree and return chunks
        tree = etree.parse(io.BytesIO(content))
        root = tree.getroot()

        framework_chunks: List[Document] = []
        dg_chunks = get_leaf_structural_chunks(
            root,
            min_chunk_size=self.min_chunk_size,
            whitespace_normalize_text=self.whitespace_normalize_text,
            sub_chunk_tables=self.sub_chunk_tables,
            include_xml_tags=self.include_xml_tags,
            xml_hierarchy_levels=self.xml_hierarchy_levels,
            max_text_size=self.max_text_length,
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

            if artifact_name == "report-values.xml" and artifact_url and artifact_doc:
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

    from langchain.chat_models import ChatOpenAI

    def get_chain(
        self,
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
    ):
        if not self.access_token and self.docset_id:
            raise Exception(f"Please specify a docset ID to use agent executor")

        from langchain.sql_database import SQLDatabase
        from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
        from langchain.agents.agent_toolkits.sql.base import create_sql_agent
        from langchain.agents.agent_types import AgentType

        import sqlite3
        import pandas as pd
        import tempfile

        response = requests.request(
            "GET",
            f"{self.api}/projects/?docset.id={self.docset_id}",
            headers={"Authorization": f"Bearer {self.access_token}"},
            data={},
        )
        if response.ok:
            response_json = response.json()
            projects = response_json["projects"]
            if not projects or not projects[0]["id"]:
                raise Exception(
                    f"Could not access published projects for docset: {self.docset_id}"
                )

            project_id = projects[0]["id"]
            project_name = projects[0].get("name") or "Report"
            response = requests.request(
                "GET",
                f"{self.api}/projects/{project_id}/artifacts/latest?name=spreadsheet.xlsx",
                headers={"Authorization": f"Bearer {self.access_token}"},
                data={},
            )
            if response.ok:
                response_json = response.json()
                artifacts = response_json["artifacts"]
                if not artifacts or not artifacts[0]["downloadUrl"]:
                    raise Exception(
                        f"Could not find download URL for latest artifact in project: {project_id}"
                    )

                download_url = artifacts[0]["downloadUrl"]

                response = requests.request(
                    "GET",
                    download_url,
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    data={},
                )
                if response.ok:
                    temp_xlsx_file = tempfile.NamedTemporaryFile(
                        suffix=".xlsx", delete=False
                    )
                    temp_xlsx_path = temp_xlsx_file.name
                    temp_xlsx_file.close()

                    with open(temp_xlsx_path, "wb") as f:
                        f.write(response.content)
                        f.flush()

                    # Create a temporary SQLite database in memory
                    in_memory_sqlite_connection = sqlite3.connect(":memory:")

                    # Read the Excel file using pandas (only the first sheet)
                    df = pd.read_excel(temp_xlsx_path, sheet_name=0)

                    # Write the Excel file from pandas to the in-memory sqlite connection
                    df.to_sql(
                        project_name,
                        in_memory_sqlite_connection,
                        if_exists="replace",
                        index=False,
                    )

                    # Dump the in-memory sqlite connection to a db file on disk
                    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite")
                    with sqlite3.connect(temp_db_file.name) as disk_conn:
                        in_memory_sqlite_connection.backup(disk_conn)

                    # Connect to the db file on disk
                    db = SQLDatabase.from_uri(
                        f"sqlite:///{temp_db_file.name}", sample_rows_in_table_info=3
                    )

                    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
                    return create_sql_agent(
                        llm=llm,
                        toolkit=toolkit,
                        verbose=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    )
                else:
                    raise Exception(f"Failed to download XLSX for {project_id}")
            else:
                raise Exception("Failed to get docsets.")
