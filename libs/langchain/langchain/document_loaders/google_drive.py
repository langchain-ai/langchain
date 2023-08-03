import itertools
import logging
import os
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
)

from pydantic.class_validators import root_validator

from langchain.base_language import BaseLanguageModel
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

from ..utilities.google_drive import (
    GoogleDriveUtilities,
    get_template,
)


# To manage a circular import, use an alias of PromptTemplate
class PromptTemplate(Protocol):
    input_variables: List[str]
    template: str

    def format(self, **kwargs: Any) -> str:
        ...


logger = logging.getLogger(__name__)


class GoogleDriveLoader(BaseLoader, GoogleDriveUtilities):
    document_ids: Optional[List[str]] = None  # deprecated
    """ A list of google drive document ids to load."""

    file_ids: Optional[List[str]] = None  # deprecated
    """ A list of google drive file ids to load."""

    @root_validator(pre=True)
    def validate_older_api_and_new_environment_variable(
        cls, v: Dict[str, Any]
    ) -> Dict[str, Any]:
        from langchain import PromptTemplate as OriginalPromptTemplate

        service_account_key = v.get("service_account_key")
        credentials_path = v.get("credentials_path")
        api_file = v.get("gdrive_api_file")

        if service_account_key:
            warnings.warn(
                "service_account_key was deprecated. Use GOOGLE_ACCOUNT_FILE env. "
                "variable.",
                DeprecationWarning,
            )
        if credentials_path:
            warnings.warn(
                "service_account_key was deprecated. Use GOOGLE_ACCOUNT_FILE env. "
                "variable.",
                DeprecationWarning,
            )
        if service_account_key and credentials_path:
            raise ValueError("Select only service_account_key or service_account_key")

        folder_id = v.get("folder_id")
        document_ids = v.get("document_ids")
        file_ids = v.get("file_ids")

        if folder_id and (document_ids or file_ids):
            raise ValueError(
                "Cannot specify both folder_id and document_ids nor "
                "folder_id and file_ids"
            )

        # To be compatible with the old approach
        if not api_file:
            api_file = (
                Path(os.environ["GOOGLE_ACCOUNT_FILE"])
                if "GOOGLE_ACCOUNT_FILE" in os.environ
                else None
            )
            # Deprecated: To be compatible with the old approach of authentication
            if service_account_key:
                api_file = service_account_key
            elif credentials_path:
                api_file = credentials_path
            elif not api_file:
                api_file = Path.home() / ".credentials" / "keys.json"
            v["gdrive_api_file"] = api_file

        if not v.get("template"):
            if folder_id:
                template = get_template("gdrive-all-in-folder")
            elif "document_ids" in v or "file_ids" in v:
                template = OriginalPromptTemplate(input_variables=[], template="")
            else:
                raise ValueError("Use a template")
            v["template"] = template
        return v

    def _lazy_load_documents_ids(self) -> Iterator[Document]:
        if not self.document_ids:
            return iter([])
        return itertools.chain.from_iterable(
            [self.lazy_load_document_from_id(doc_id) for doc_id in self.document_ids]
        )

    def _lazy_load_files_ids(self) -> Iterator[Document]:
        if not self.file_ids:
            return iter([])
        return itertools.chain.from_iterable(
            [self.lazy_load_file_from_id(doc_id) for doc_id in self.file_ids]
        )

    def lazy_get_relevant_documents(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> Iterator[Document]:
        if self.document_ids:
            return self._lazy_load_documents_ids()
        if self.file_ids:
            return self._lazy_load_files_ids()
        return super().lazy_get_relevant_documents(query=query, **kwargs)

    def lazy_load(self) -> Iterator[Document]:
        return self.lazy_get_relevant_documents()

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_update_description_with_summary(
        self,
        llm: BaseLanguageModel,
        force: bool = False,
        prompt: Optional[PromptTemplate] = None,
        query: str = "",
        **kwargs: Any,
    ) -> Iterator[Document]:
        """Summarize all documents, and update the GDrive metadata `description`.

        Need `write` access: set scope=["https://www.googleapis.com/auth/drive"].

        Note: Update the description of shortcut without touch the target
        file description.

        Args:
            llm: Language modele to use.
            force: true to update all files. Else, update only if the description
                is empty.
            query: If possible, the query request.
            kwargs: Others parameters for the template (verbose, prompt, etc).
        """
        from googleapiclient.errors import HttpError

        from langchain.chains.summarize import load_summarize_chain

        if "https://www.googleapis.com/auth/drive" not in self._creds.scopes:
            raise ValueError(
                f"Remove the file 'token.json' and "
                f"initialize the {self.__class__.__name__} with "
                f"scopes=['https://www.googleapis.com/auth/drive']"
            )

        chain = load_summarize_chain(llm, chain_type="stuff", **kwargs)
        updated_files = set()  # Never update two time the same document (if it's split)
        for document in self.lazy_get_relevant_documents(query, **kwargs):
            try:
                file_id = document.metadata["gdriveId"]
                if file_id not in updated_files:
                    file = self.files.get(
                        fileId=file_id,
                        fields=self.fields,
                        supportsAllDrives=True,
                    ).execute()
                    if force or not file.get("description", "").strip():
                        summary = chain.run([document]).strip()
                        if summary:
                            self.files.update(
                                fileId=file_id,
                                supportsAllDrives=True,
                                body={"description": summary},
                            ).execute()
                            logger.info(
                                f"For the file '{file['name']}', add description "
                                f"'{summary[:40]}...'"
                            )
                            metadata = self._extract_meta_data(file)
                            if "summary" in metadata:
                                del metadata["summary"]
                            yield Document(page_content=summary, metadata=metadata)
                    updated_files.add(file_id)
            except HttpError:
                logger.warning(
                    f"Impossible to update the description of file "
                    f"'{document.metadata['name']}'"
                )

    def update_description_with_summary(
        self,
        llm: BaseLanguageModel,
        force: bool = False,
        query: str = "",
        **kwargs: Any,
    ) -> List[Document]:
        """Summarize all documents, and update the GDrive metadata `description`.

        Need `write` access: set scope=["https://www.googleapis.com/auth/drive"].

        Note: Update the description of shortcut without touch the target
        file description.

        Args:
            llm: Language modele to use.
            force: true to update all files. Else, update only if the description
                is empty.
            query: If possible, the query request.
            kwargs: Others parameters for the template (verbose, prompt, etc).
        """
        return list(
            self.lazy_update_description_with_summary(
                llm=llm, force=force, query=query, **kwargs
            )
        )
