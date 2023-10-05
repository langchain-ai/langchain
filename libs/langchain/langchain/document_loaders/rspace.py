import os

from pydantic import root_validator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.pydantic_v1 import BaseModel
from typing import Any, Dict, List, Optional, Iterator, Union
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.utils import get_from_dict_or_env

load_dotenv()


class RSpaceLoader(BaseLoader, BaseModel):
    """ Load files, notebooks, folders or documents from RSpace.

    Requires an RSpace API key which is obtainable from user's profile page
    Also, needs RSpace Python SDK (`pip install rspace_client`)

    RSpace documents map 1-1 to Langchain documents.

    PDF attachments in the Gallery are parsed by PyPDFLoader (requires pypdf)

    For more details of RSpace Electronic Lab Notebook please see https://www.researchspace.com

    """

    api_key: str
    """RSpace API key - can be supplied as environment variable 'RSPACE_API_KEY' """

    url: str
    """ The URL of your RSpace instance - can be supplied as environment variable 'RSPACE_URL'"""

    global_id: str
    """ The global ID of the resource to load, 
    e.g. 'SD12344' (a single document); 'GL12345'(A PDF file in the gallery); 'NB4567' (a notebook);
     'FL12244' (a folder)
    """

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that API key and URL exists in environment."""
        values["api_key"] = get_from_dict_or_env(
            values, "api_key", "RSPACE_API_KEY"
        )
        values["url"] = get_from_dict_or_env(
            values, "url", "RSPACE_URL"
        )
        if 'global_id' not in values or values['global_id'] is None:
            raise ValueError("No value supplied for global_id. Please supply an RSpace global ID")
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
                f"Unable to initialise client - is url {self.url} or key of length {len(self.api_key)} correct?")

        return eln, field_content.FieldContent

    def _get_doc(self, cli, field_content, d_id):
        content = ""
        doc = cli.get_document(d_id)
        content += f"<h2>{doc['name']}<h2/>"
        for f in doc['fields']:
            content += f"{f['name']}\n"
            fc = field_content(f['content'])
            content += fc.get_text()
            content += '\n'
        return Document(metadata={'source': f"rspace: {doc['name']}-{doc['globalId']}"}, page_content=content)

    def _load_structured_doc(self) -> Iterator[Document]:
        cli, field_content = self._create_rspace_client()
        yield self._get_doc(cli, field_content, self.global_id)

    def _load_folder_tree(self) -> Iterator[Document]:
        cli, field_content = self._create_rspace_client()
        docs_in_folder = cli.list_folder_tree(folder_id=self.global_id[2:], typesToInclude=['document'])
        doc_ids = [d['id'] for d in docs_in_folder['records']]
        for doc_id in doc_ids:
            yield self._get_doc(cli, field_content, doc_id)

    def _load_pdf(self) -> Iterator[Document]:
        cli, field_content = self._create_rspace_client()
        file_info = cli.get_file_info(self.global_id)
        print(file_info)
        _, ext = os.path.splitext(file_info['name'])
        if ext.lower() == '.pdf':
            outfile = f"{self.global_id}.pdf"
            cli.download_file(self.global_id, outfile)
            pdf_loader = PyPDFLoader(outfile)
            for pdf in pdf_loader.lazy_load():
                pdf.metadata['rspace_src'] = self.global_id
                yield pdf

    def lazy_load(self) -> Iterator[Document]:
        if 'GL' in self.global_id:
            for d in self._load_pdf():
                yield d
        elif 'SD' in self.global_id:
            for d in self._load_structured_doc():
                yield d
        elif self.global_id[0:2] in ['FL, NB']:
            for d in self._load_folder_tree():
                yield d
        else:
            raise ValueError("Unknown global ID type")

    def load(self) -> List[Document]:
        return list(self.lazy_load())