import os
from typing import Any, Dict, Iterator, List, Optional, Union

from langchain_core.documents import Document

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.base import BaseLoader
from langchain.utils import get_from_dict_or_env


class RSpaceLoader(BaseLoader):
    """
    Loads  content from RSpace notebooks, folders, documents or PDF Gallery files into
    Langchain documents.

    Maps  RSpace document <-> Langchain Document in 1-1. PDFs are imported using PyPDF.

    Requirements are rspace_client (`pip install rspace_client`) and PyPDF if importing
     PDF docs (`pip install pypdf`).

    """

    def __init__(
        self, global_id: str, api_key: Optional[str] = None, url: Optional[str] = None
    ):
        """api_key: RSpace API key - can also be supplied as environment variable
        'RSPACE_API_KEY'
        url: str
        The URL of your RSpace instance - can also be supplied as environment
        variable 'RSPACE_URL'
        global_id: str
         The global ID of the resource to load,
        e.g. 'SD12344' (a single document); 'GL12345'(A PDF file in the gallery);
        'NB4567' (a notebook); 'FL12244' (a folder)
        """
        args: Dict[str, Optional[str]] = {
            "api_key": api_key,
            "url": url,
            "global_id": global_id,
        }
        verified_args: Dict[str, str] = RSpaceLoader.validate_environment(args)
        self.api_key = verified_args["api_key"]
        self.url = verified_args["url"]
        self.global_id: str = verified_args["global_id"]

    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that API key and URL exists in environment."""
        values["api_key"] = get_from_dict_or_env(values, "api_key", "RSPACE_API_KEY")
        values["url"] = get_from_dict_or_env(values, "url", "RSPACE_URL")
        if "global_id" not in values or values["global_id"] is None:
            raise ValueError(
                "No value supplied for global_id. Please supply an RSpace global ID"
            )
        return values

    def _create_rspace_client(self) -> Any:
        """Create a RSpace client."""
        try:
            from rspace_client.eln import eln, field_content

        except ImportError:
            raise ImportError("You must run " "`pip install rspace_client`")

        try:
            eln = eln.ELNClient(self.url, self.api_key)
            eln.get_status()

        except Exception:
            raise Exception(
                f"Unable to initialise client - is url {self.url} or "
                f"api key  correct?"
            )

        return eln, field_content.FieldContent

    def _get_doc(self, cli: Any, field_content: Any, d_id: Union[str, int]) -> Document:
        content = ""
        doc = cli.get_document(d_id)
        content += f"<h2>{doc['name']}<h2/>"
        for f in doc["fields"]:
            content += f"{f['name']}\n"
            fc = field_content(f["content"])
            content += fc.get_text()
            content += "\n"
        return Document(
            metadata={"source": f"rspace: {doc['name']}-{doc['globalId']}"},
            page_content=content,
        )

    def _load_structured_doc(self) -> Iterator[Document]:
        cli, field_content = self._create_rspace_client()
        yield self._get_doc(cli, field_content, self.global_id)

    def _load_folder_tree(self) -> Iterator[Document]:
        cli, field_content = self._create_rspace_client()
        if self.global_id:
            docs_in_folder = cli.list_folder_tree(
                folder_id=self.global_id[2:], typesToInclude=["document"]
            )
        doc_ids: List[int] = [d["id"] for d in docs_in_folder["records"]]
        for doc_id in doc_ids:
            yield self._get_doc(cli, field_content, doc_id)

    def _load_pdf(self) -> Iterator[Document]:
        cli, field_content = self._create_rspace_client()
        file_info = cli.get_file_info(self.global_id)
        _, ext = os.path.splitext(file_info["name"])
        if ext.lower() == ".pdf":
            outfile = f"{self.global_id}.pdf"
            cli.download_file(self.global_id, outfile)
            pdf_loader = PyPDFLoader(outfile)
            for pdf in pdf_loader.lazy_load():
                pdf.metadata["rspace_src"] = self.global_id
                yield pdf

    def lazy_load(self) -> Iterator[Document]:
        if self.global_id and "GL" in self.global_id:
            for d in self._load_pdf():
                yield d
        elif self.global_id and "SD" in self.global_id:
            for d in self._load_structured_doc():
                yield d
        elif self.global_id and self.global_id[0:2] in ["FL", "NB"]:
            for d in self._load_folder_tree():
                yield d
        else:
            raise ValueError("Unknown global ID type")

    def load(self) -> List[Document]:
        return list(self.lazy_load())
