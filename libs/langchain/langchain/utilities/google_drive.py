import io
import json
import logging
import mimetypes
import os
import re
import tempfile
import traceback
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Type,
    Union,
    cast,
    runtime_checkable,
)
from uuid import UUID, uuid4

from pydantic import root_validator
from pydantic.class_validators import validator
from pydantic.config import Extra
from pydantic.fields import Field
from pydantic.main import BaseModel
from pydantic.types import FilePath

from langchain.load.serializable import Serializable
from langchain.schema import Document

# from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter


# BaseLoader=Any # Fix circular import
class BaseLoader(Protocol):
    """Interface for loading Documents.

    Implementations should implement the lazy-loading method using generators
    to avoid loading all Documents into memory at once.

    The `load` method will remain as is for backwards compatibility, but its
    implementation should be just `list(self.lazy_load())`.
    """

    # Sub-classes should implement this method
    # as return list(self.lazy_load()).
    # This method returns a List which is materialized in memory.
    def load(self) -> List[Document]:
        ...

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        ...

    # Attention: This method will be upgraded into an abstractmethod once it's
    #            implemented in all the existing subclasses.
    def lazy_load(
        self,
    ) -> Iterator[Document]:
        ...


FORMAT_INSTRUCTION = (
    "The input should be formatted as a list of entities separated"
    " with a space. As an example, a list of keywords is 'hello word'."
)


@runtime_checkable
class _FilePathLoader(Protocol):
    def __call__(self, file_path: str, **kwargs: Dict[str, Any]) -> BaseLoader:
        ...


@runtime_checkable
class _FilePathLoaderProtocol(Protocol):
    def __init__(self, file_path: str, **kwargs: Dict[str, Any]):
        ...

    def load(self) -> List[Document]:
        ...


TYPE_CONV_MAPPING = Dict[str, Union[_FilePathLoader, Type[_FilePathLoaderProtocol]]]

_acceptable_params_of_list = {
    "corpora",
    "driveId",
    "fields",
    "includeItemsFromAllDrives",
    "orderBy",
    "pageSize",
    "pageToken",
    "q",
    "spaces",
    "supportsAllDrives",
    "includePermissionsForView",
    "includeLabels",
}


# To manage a circular import, use an alias of PromptTemplate
@runtime_checkable
class PromptTemplate(Protocol):
    input_variables: List[str]
    template: str

    def format(self, **kwargs: Any) -> str:
        ...


logger = logging.getLogger(__name__)

# Manage :
# - File in trash
# - Shortcut
# - Paging with request GDrive list()
# - Multiple kind of template for request GDrive
# - Convert a lot of mime type (can be configured)
# - Convert GDoc, GSheet and GSlide
# - Can use only the description of files, without conversion of the body
# - Lambda filter
# - Remove duplicate document (via shortcut)
# - All GDrive api parameters
# - Url to documents
# - Environment variable for reference the API tokens
# - Different kind of strange state with Google File (absence of URL, etc.)

SCOPES: List[str] = [
    # See https://developers.google.com/identity/protocols/oauth2/scopes
    "https://www.googleapis.com/auth/drive.readonly",
]


class _LRUCache:
    # initialising capacity
    def __init__(self, capacity: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._capacity: int = capacity

    def get(self, key: str) -> Optional[str]:
        if key not in self._cache:
            return None
        else:
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)


def default_conv_loader(
    mode: Literal["single", "elements"] = "single",
    strategy: Literal["strategy", "fast"] = "fast",
    ocr_languages: str = "eng",
) -> TYPE_CONV_MAPPING:
    mime_types_mapping: TYPE_CONV_MAPPING = {}
    try:
        from langchain.document_loaders import TextLoader

        mime_types_mapping.update(
            {
                "text/text": TextLoader,
                "text/plain": TextLoader,
            }
        )
    except ImportError:
        # Ignore TextLoader
        logger.info("Ignore TextLoader for GDrive")

    try:
        from langchain.document_loaders import CSVLoader

        mime_types_mapping.update(
            {
                "text/csv": CSVLoader,
            }
        )
    except ImportError:
        # Ignore CVS
        logger.info("Ignore CSVLoader for GDrive")

    try:
        from langchain.document_loaders import NotebookLoader

        mime_types_mapping.update(
            {
                "application/vnd.google.colaboratory": partial(
                    lambda file_path: NotebookLoader(
                        path=file_path, include_outputs=False, remove_newline=True
                    )
                ),  # Notebooks
            }
        )
    except ImportError:
        logger.info("Ignore NotebookLoader for GDrive")

    try:
        import pypandoc

        from langchain.document_loaders import UnstructuredRTFLoader

        pypandoc.ensure_pandoc_installed()

        mime_types_mapping.update(
            {
                "application/rtf": UnstructuredRTFLoader,
            }
        )
    except ImportError:
        logger.info("Ignore RTF for GDrive (use `pip install pypandoc_binary`)")
    try:
        import unstructured  # noqa: F401

        from langchain.document_loaders import (
            UnstructuredEPubLoader,
            UnstructuredFileLoader,
            UnstructuredHTMLLoader,
            UnstructuredImageLoader,
            UnstructuredMarkdownLoader,
            UnstructuredODTLoader,
            UnstructuredPDFLoader,
            UnstructuredPowerPointLoader,
            UnstructuredWordDocumentLoader,
        )

        try:
            import detectron2  # noqa: F401
            import pdf2image  # noqa: F401
            import pytesseract

            mime_types_mapping.update(
                {
                    "image/png": partial(
                        UnstructuredImageLoader, ocr_languages=ocr_languages
                    ),
                    "image/jpeg": partial(
                        UnstructuredImageLoader, ocr_languages=ocr_languages
                    ),
                    "application/json": partial(
                        UnstructuredFileLoader, ocr_languages=ocr_languages
                    ),
                }
            )
        except ImportError:
            logger.info(
                "Ignore Images for GDrive (no module named "
                "'pdf2image', 'detectron2' and 'pytesseract')"
            )

        try:
            import pypandoc  # noqa: F401, F811

            mime_types_mapping.update(
                {
                    "application/epub+zip": UnstructuredEPubLoader,
                }
            )
        except ImportError:
            logger.info("Ignore Epub for GDrive (no module named 'pypandoc'")

        try:
            import pdf2image  # noqa: F401, F811
            import pytesseract  # noqa: F401, F811

            mime_types_mapping.update(
                {
                    "application/pdf": partial(
                        UnstructuredPDFLoader, strategy=strategy, mode=mode
                    ),
                }
            )
        except ImportError:
            logger.info(
                "Ignore PDF for GDrive (no module named 'pdf2image' "
                "and 'pytesseract'"
            )

        mime_types_mapping.update(
            {
                "text/html": UnstructuredHTMLLoader,
                "text/markdown": UnstructuredMarkdownLoader,
                "application/vnd.openxmlformats-officedocument."
                "presentationml.presentation": partial(
                    UnstructuredPowerPointLoader, mode=mode
                ),  # PPTX
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document": partial(
                    UnstructuredWordDocumentLoader, mode=mode
                ),  # DOCX
                # "application/vnd.openxmlformats-officedocument.
                # spreadsheetml.sheet": # XLSX
                "application/vnd.oasis.opendocument.text": UnstructuredODTLoader,
            }
        )
    except ImportError:
        logger.info(
            "Ignore Unstructure*Loader for GDrive "
            "(no module `unstructured[local-inference]`)"
        )

    return mime_types_mapping


def _init_templates() -> Dict[str, PromptTemplate]:
    from langchain.prompts.prompt import PromptTemplate as MyPromptTemplate

    return {
        "gdrive-all-in-folder": MyPromptTemplate(
            input_variables=["folder_id"],
            template=" '{folder_id}' in parents and trashed=false",
        ),
        "gdrive-query": MyPromptTemplate(
            input_variables=["query"],
            template="fullText contains '{query}' and trashed=false",
        ),
        "gdrive-by-name": MyPromptTemplate(
            input_variables=["query"],
            template="name contains '{query}' and trashed=false",
        ),
        "gdrive-by-name-in-folder": MyPromptTemplate(
            input_variables=["query", "folder_id"],
            template="name contains '{query}' "
            "and '{folder_id}' in parents "
            "and trashed=false",
        ),
        "gdrive-query-in-folder": MyPromptTemplate(
            input_variables=["query", "folder_id"],
            template="fullText contains '{query}' "
            "and '{folder_id}' in parents "
            "and trashed=false",
        ),
        "gdrive-mime-type": MyPromptTemplate(
            input_variables=["mime_type"],
            template="mimeType = '{mime_type}' and trashed=false",
        ),
        "gdrive-mime-type-in-folder": MyPromptTemplate(
            input_variables=["mime_type", "folder_id"],
            template="mimeType = '{mime_type}' "
            "and '{folder_id}' in parents "
            "and trashed=false",
        ),
        "gdrive-query-with-mime-type": MyPromptTemplate(
            input_variables=["query", "mime_type"],
            template="(fullText contains '{query}' "
            "and mime_type = '{mime_type}') "
            "and trashed=false",
        ),
        "gdrive-query-with-mime-type-and-folder": MyPromptTemplate(
            input_variables=["query", "mime_type", "folder_id"],
            template="((fullText contains '{query}') and mime_type = '{mime_type}')"
            "and '{folder_id}' in parents "
            "and trashed=false",
        ),
    }


templates: Optional[Dict[str, PromptTemplate]] = None


def get_template(template: str) -> PromptTemplate:
    global templates
    if not templates:
        templates = _init_templates()
    return templates[template]


class GoogleDriveUtilities(Serializable, BaseModel):
    """
    Loader that loads documents from Google Drive.

    All files that can be converted to text can be converted to `Document`.
    - All documents use the `conv_mapping` to extract the text.

    At this time, the default list of accepted mime-type is:
    - text/text
    - text/plain
    - text/html
    - text/csv
    - text/markdown
    - image/png
    - image/jpeg
    - application/epub+zip
    - application/pdf
    - application/rtf
    - application/vnd.google-apps.document (GDoc)
    - application/vnd.google-apps.presentation (GSlide)
    - application/vnd.google-apps.spreadsheet (GSheet)
    - application/vnd.google.colaboratory (Notebook colab)
    - application/vnd.openxmlformats-officedocument.presentationml.presentation (PPTX)
    - application/vnd.openxmlformats-officedocument.wordprocessingml.document (DOCX)

    All empty files are ignored.

    The code use the Google API v3. To have more information about some parameters,
    see [here](https://developers.google.com/drive/api/v3/reference/files/list).

    The application must be authenticated with a json file.
    The format may be for a user or for an application via a service account.
    The environment variable `GOOGLE_ACCOUNT_FILE` may be set to reference this file.
    For more information, see [here]
    (https://developers.google.com/workspace/guides/auth-overview).

    All parameter compatible with Google [`list()`]
    (https://developers.google.com/drive/api/v3/reference/files/list)
    API can be set.

    To specify the new pattern of the Google request, you can use a `PromptTemplate()`.
    The variables for the prompt can be set with `kwargs` in the constructor.
    Some pre-formated request are proposed (use {query}, {folder_id}
    and/or {mime_type}):
    - "gdrive-all-in-folder":                   Return all compatible files from a
                                                 `folder_id`
    - "gdrive-query":                           Search `query` in all drives
    - "gdrive-by-name":                         Search file with name `query`)
    - "gdrive-by-name-in-folder":               Search file with name `query`)
                                                 in `folder_id`
    - "gdrive-query-in-folder":                 Search `query` in `folder_id`
                                                 (and sub-folders in `recursive=true`)
    - "gdrive-mime-type":                       Search a specific `mime_type`
    - "gdrive-mime-type-in-folder":             Search a specific `mime_type` in
                                                 `folder_id`
    - "gdrive-query-with-mime-type":            Search `query` with a specific
                                                 `mime_type`
    - "gdrive-query-with-mime-type-and-folder": Search `query` with a specific
                                                 `mime_type` and in `folder_id`

    If you ask to use only the `description` of each file (mode='snippets'):
    - If a link has a description, use it
    - Else, use the description of the target_id file
    - If the description is empty, ignore the file
    ```
    Sample of use:
    documents = list(GoogleDriveUtilities(
                gdrive_api_file=os.environ["GOOGLE_ACCOUNT_FILE"],
                num_results=10,
                template="gdrive-query-in-folder",
                recursive=True,
                filter=lambda search, file: "#ai" in file.get('description',''),
                folder_id='root',
                query='LLM',
                supportsAllDrives=False,
                ).lazy_get_relevant_documents())
    ```
    """

    class Config:
        extra = Extra.allow
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True
        allow_mutation = True  # deprecated. Only for back compatibility
        # validate_assignment = True  # deprecated. Only for back compatibility

    @property
    def files(self) -> Any:
        """Google workspakce files interface"""
        return self._files

    gdrive_api_file: Optional[FilePath]
    """
    The file to use to connect to the google api or use 
    `os.environ["GOOGLE_ACCOUNT_FILE"]`. May be a user or service json file"""

    not_data = uuid4()

    @validator("gdrive_api_file", always=True)
    def validate_api_file(cls, api_file: Optional[FilePath]) -> FilePath:
        if not api_file:
            env_api_file = os.environ.get("GOOGLE_ACCOUNT_FILE")
            if not env_api_file:
                raise ValueError("set GOOGLE_ACCOUNT_FILE environment variable")
            else:
                api_file = Path(env_api_file)
        else:
            if api_file is None:
                raise ValueError("gdrive_api_file must be set")
        if not api_file.exists():
            raise ValueError(f"Api file '{api_file}' does not exist")
        return api_file

    @root_validator
    def validate_template(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        template = values.get("template")
        if isinstance(template, str):
            template = get_template(template)

        values["template"] = template
        return values

    @root_validator(pre=True)
    def validate_file_loader_cls(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("file_loader_cls") or values.get("file_loader_kwargs"):
            warnings.warn(
                "file_loader_cls and file_loader_kwargs "
                "are deprecated. Use conv_mapping.",
                DeprecationWarning,
            )
            logger.warning(
                "file_loader_cls and file_loader_kwargs "
                "are deprecated. Use conv_mapping."
            )
        return values

    gdrive_token_path: Optional[Path] = None
    """ Path to save the token.json file. By default, use the directory of 
    `gdrive_api_file."""

    num_results: int = -1
    """Number of documents to be returned by the retriever (default: -1 for all)."""

    mode: str = "documents"
    """Return the document."""

    recursive: bool = False
    """If `true`, search in the `folder_id` and sub folders."""

    filter: Callable[["GoogleDriveUtilities", Dict], bool] = cast(
        Callable[["GoogleDriveUtilities", Dict], bool], lambda self, file: True
    )
    """ A lambda/function to add some filter about the google file item."""

    link_field: Literal["webViewLink", "webContentLink"] = "webViewLink"
    """Google API return two url for the same file.
      `webViewLink` is to open the document inline, and `webContentLink` is to
      download the document. Select the field to use for the documents."""

    follow_shortcut: bool = True
    """If `true` and find a google link to document or folder, follow it."""

    conv_mapping: TYPE_CONV_MAPPING = Field(default_factory=default_conv_loader)
    """A dictionary to map a mime-type and a loader"""

    gslide_mode: Literal["single", "elements", "slide"] = "single"
    """Generate one document by slide,
            one document with <PAGE BREAK> (`single`),
            one document by slide (`slide`)
            or one document for each `elements`."""

    gsheet_mode: Literal["single", "elements"] = "single"
    """Generate one document by line ("single"),
            or one document with markdown array and `<PAGE BREAK>` tags."""

    scopes: List[str] = SCOPES
    """ The scope to use the Google API. The default is for Read-only. 
    See [here](https://developers.google.com/identity/protocols/oauth2/scopes) """

    # Google Drive parameters
    corpora: Optional[Literal["user", "drive", "domain", "allDrives"]] = None
    """
    Groupings of files to which the query applies.
    Supported groupings are: 'user' (files created by, opened by, or shared directly 
    with the user),
    'drive' (files in the specified shared drive as indicated by the 'driveId'),
    'domain' (files shared to the user's domain), and 'allDrives' (A combination of 
    'user' and 'drive' for all drives where the user is a member).
    When able, use 'user' or 'drive', instead of 'allDrives', for efficiency."""

    driveId: Optional[str] = None
    """ID of the shared drive to search."""

    fields: str = (
        "id, name, mimeType, description, webViewLink, "
        "webContentLink, owners/displayName, shortcutDetails, "
        "sha256Checksum, modifiedTime"
    )
    """The paths of the fields you want included in the response.
        If not specified, the response includes a default set of fields specific to this
        method.
        For development, you can use the special value * to return all fields, but 
        you'll achieve greater performance by only selecting the fields you need. For 
        more information, see [Return specific fields for a file]
        (https://developers.google.com/drive/api/v3/fields-parameter)."""

    includeItemsFromAllDrives: Optional[bool] = False
    """Whether both My Drive and shared drive items should be included in results."""

    includeLabels: Optional[bool] = None
    """A comma-separated list of IDs of labels to include in the labelInfo part of 
    the response."""

    includePermissionsForView: Optional[Literal["published"]] = None
    """Specifies which additional view's permissions to include in the response.
    Only 'published' is supported."""

    orderBy: Optional[
        Literal[
            "createdTime",
            "folder",
            "modifiedByMeTime",
            "modifiedTime",
            "name",
            "name_natural",
            "quotaBytesUsed",
            "recency",
            "sharedWithMeTime",
            "starred",
            "viewedByMeTime",
        ]
    ] = None
    """
    A comma-separated list of sort keys. Valid keys are 'createdTime', 'folder', 
    'modifiedByMeTime', 'modifiedTime', 'name', 'name_natural', 'quotaBytesUsed', 
    'recency', 'sharedWithMeTime', 'starred', and 'viewedByMeTime'. Each key sorts 
    ascending by default, but may be reversed with the 'desc' modifier. 
    Example usage: `orderBy="folder,modifiedTime desc,name"`. Please note that there is
    a current limitation for users with approximately one million files in which the 
    requested sort order is ignored."""

    pageSize: int = 100
    """
    The maximum number of files to return per page. Partial or empty result pages are
    possible even before the end of the files list has been reached. Acceptable 
    values are 1 to 1000, inclusive."""

    spaces: Optional[Literal["drive", "appDataFolder"]] = None
    """A comma-separated list of spaces to query within the corpora. Supported values 
    are `drive` and `appDataFolder`."""

    supportsAllDrives: bool = True
    """Whether the requesting application supports both My Drives and
                shared drives. (Default: true)"""

    file_loader_cls: Optional[Type] = None  # deprecated
    """Deprecated: The file loader class to use."""
    file_loader_kwargs: Optional[Dict[str, Any]] = None  # deprecated
    """Deprecated: The file loader kwargs to use."""

    template: Union[
        PromptTemplate,
        Literal[
            "gdrive-all-in-folder",
            "gdrive-query",
            "gdrive-by-name",
            "gdrive-by-name-in-folder",
            "gdrive-query-in-folder",
            "gdrive-mime-type",
            "gdrive-mime-type-in-folder",
            "gdrive-query-with-mime-type",
            "gdrive-query-with-mime-type-and-folder",
        ],
        None,
    ] = None
    """
    A `PromptTemplate` with the syntax compatible with the parameter `q` 
    of Google API').
    The variables may be set in the constructor, or during the invocation of 
    `lazy_get_relevant_documents()`.
    """

    # Private fields
    _files = Field(allow_mutation=True)
    _docs = Field(allow_mutation=True)
    _spreadsheets = Field(allow_mutation=True)
    _slides = Field(allow_mutation=True)
    _creds = Field(allow_mutation=True)
    _gdrive_kwargs: Dict[str, Any] = Field(allow_mutation=True)
    _kwargs: Dict[str, Any] = Field(allow_mutation=True)
    _folder_name_cache: _LRUCache = Field(default_factory=_LRUCache)
    _not_supported: Set = Field(default_factory=set)
    _no_data: UUID = Field(default_factory=uuid4)

    # Class var
    _default_page_size: ClassVar[int] = 50

    @root_validator
    def orderBy_is_compatible_with_recursive(
        cls, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        if values["orderBy"] and values["recursive"]:
            raise ValueError("`orderBy` is incompatible with `recursive` parameter")
        return values

    _gdrive_list_params: ClassVar[Set[str]] = {
        "corpora",
        "corpus",
        "driveId",
        "fields",
        "includeItemsFromAllDrives",
        "includeLabels",
        "includePermissionsForView",
        "includeTeamDriveItems",
        "orderBy",
        "pageSize",
        "pageToken",
        "q",
        "spaces",
        "supportsAllDrives",
        "supportsTeamDrives",
        "teamDriveId",
    }
    _gdrive_get_params: ClassVar[Set[str]] = {
        "id",
        "acknowledgeAbuse",
        "fields",
        "includeLabels",
        "includePermissionsForView",
        "supportsAllDrives",
        "supportsTeamDrives",
    }

    def _load_credentials(self, api_file: Optional[Path], scopes: List[str]) -> Any:
        """Load credentials.

         Args:
            api_file: The user or services json file

        Returns:
            credentials.
        """
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "google-api-python-client google-auth-httplib2 "
                "google-auth-oauthlib` "
                "to use the Google Drive loader."
            )

        if api_file:
            with io.open(api_file, "r", encoding="utf-8-sig") as json_file:
                data = json.load(json_file)
            if "installed" in data:
                credentials_path = api_file
                service_account_key = None
            else:
                service_account_key = api_file
                credentials_path = None
        else:
            raise ValueError("Use GOOGLE_ACCOUNT_FILE env. variable.")

        # Implicit location of token.json
        if not self.gdrive_token_path and credentials_path:
            token_path = credentials_path.parent / "token.json"

        creds = None
        if service_account_key and service_account_key.exists():
            return service_account.Credentials.from_service_account_file(
                str(service_account_key), scopes=scopes
            )

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), scopes
                )
                creds = flow.run_local_server(port=0)
            with open(token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    @staticmethod
    def _snippet_from_page_content(page_content: str, max_size: int = 50) -> str:
        if max_size < 6:
            raise ValueError("max_size must be >=6")
        part_size = max_size // 2
        strip_content = re.sub(r"(\s|<PAGE BREAK>)+", r" ", page_content).strip()
        if len(strip_content) <= max_size:
            return strip_content
        elif len(strip_content) <= max_size + 3:
            return (strip_content[:part_size] + "...")[:max_size]
        return strip_content[:part_size] + "..." + strip_content[-part_size:]

    @staticmethod
    def _extract_mime_type(file: Dict[str, Any]) -> str:
        """Extract mime type or try to deduce from the filename and webViewLink"""
        if "mimeType" in file:
            mime_type = file["mimeType"]
        else:
            # Try to deduce the mime_type
            if "shortcutDetails" in file:
                return "application/vnd.google-apps.shortcut"

            suffix = Path(file["name"]).suffix
            mime_type = mimetypes.types_map.get(suffix)
            if not mime_type:
                if "webViewLink" in file:
                    match = re.search(
                        r"drive\.google\.com/drive/(.*)/", file["webViewLink"]
                    )
                    if match:
                        mime_type = "application/vnd.google-apps." + match.groups()[0]
                    else:
                        mime_type = "unknown"
                else:
                    mime_type = "unknown"
                logger.debug(
                    f"Calculate mime_type='{mime_type}' for file '{file['name']}'"
                )
        return mime_type

    def _generate_missing_url(self, file: Dict) -> Optional[str]:
        """For Google document, create the corresponding URL"""
        mime_type = file["mimeType"]
        if mime_type.startswith("application/vnd.google-apps."):
            gdrive_document_type = mime_type.split(".")[-1]
            if self.link_field == "webViewLink":
                return (
                    f"https://docs.google.com/{gdrive_document_type}/d/"
                    f"{file['id']}/edit?usp=drivesdk"
                )
            else:
                return (
                    f"https://docs.google.com/{gdrive_document_type}/uc?"
                    f"{file['id']}&export=download"
                )
        return f"https://drive.google.com/file/d/{file['id']}?usp=share_link"

    def __init__(self, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        self._template = self.template  # Deprecated.
        kwargs = {k: v for k, v in kwargs.items() if k not in self.__fields__}

        self._files = None
        self._docs = None
        self._spreadsheets = None
        self._slides = None

        self._creds = self._load_credentials(Path(self.gdrive_api_file), self.scopes)

        from googleapiclient.discovery import build

        # self._params_dict: Dict[str, Union[str, int, float]] = {}

        self._files = build("drive", "v3", credentials=self._creds).files()
        self._docs = build("docs", "v1", credentials=self._creds).documents()
        self._spreadsheets = build(
            "sheets", "v4", credentials=self._creds
        ).spreadsheets()
        self._slides = build("slides", "v1", credentials=self._creds).presentations()

        # Gdrive parameters
        self._gdrive_kwargs = {
            "corpora": self.corpora,
            "driveId": self.driveId,
            "fields": self.fields,
            "includeItemsFromAllDrives": self.includeItemsFromAllDrives,
            "includeLabels": self.includeLabels,
            "includePermissionsForView": self.includePermissionsForView,
            "orderBy": self.orderBy,
            "pageSize": self.pageSize,
            "spaces": self.spaces,
            "supportsAllDrives": self.supportsAllDrives,
        }
        # self._no_limit = False
        self._kwargs = kwargs
        self._folder_name_cache = _LRUCache()  # Cache with names of folders
        self._not_supported = set()  # Remember not supported mime type

    def get_folder_name(self, file_id: str, **kwargs: Any) -> str:
        """Return folder name from file_id. Cache the result."""
        from googleapiclient.errors import HttpError

        try:
            name = self._folder_name_cache.get(file_id)
            if name:
                return name
            else:
                name = cast(str, self._get_file_by_id(file_id)["name"])
                self._folder_name_cache.put(file_id, name)
                return name
        except HttpError:
            # Sometime, it's impossible to get the file name of a folder.
            # It's because a shortcut reference an inacessible file.
            return "inaccessible-folder"

    def _get_file_by_id(self, file_id: str, **kwargs: Any) -> Dict:
        get_kwargs = {**self._kwargs, **kwargs, **{"fields": self.fields}}
        get_kwargs = {
            key: get_kwargs[key]
            for key in get_kwargs
            if key in GoogleDriveUtilities._gdrive_get_params
        }
        return self.files.get(fileId=file_id, **get_kwargs).execute()

    def _lazy_load_file_from_file(self, file: Dict) -> Iterator[Document]:
        """
        Load document from GDrive.
        Use the `conv_mapping` dictionary to convert different kind of files.
        """
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaIoBaseDownload

        suffix = mimetypes.guess_extension(file["mimeType"])
        if not suffix:
            suffix = Path(file["name"]).suffix
        if suffix not in self._not_supported:  # Already see this suffix?
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=suffix) as tf:
                    path = tf.name
                    logger.debug(
                        f"Get '{file['name']}' with type "
                        f"'{file.get('mimeType', 'unknown')}'"
                    )

                    request = self.files.get_media(fileId=file["id"])

                    fh = io.FileIO(path, mode="wb")
                    try:
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while done is False:
                            status, done = downloader.next_chunk()

                    finally:
                        fh.close()

                    if self.file_loader_cls:
                        # Deprecated
                        request = self.files.get_media(fileId=file["id"])
                        bfh = io.BytesIO()
                        downloader = MediaIoBaseDownload(bfh, request)
                        done = False
                        while done is False:
                            status, done = downloader.next_chunk()
                        bfh.seek(0)
                        loader = self.file_loader_cls(
                            file=bfh, **self.file_loader_kwargs
                        )
                        for i, document in enumerate(loader.load()):
                            metadata = self._extract_meta_data(file)
                            if "source" in metadata:
                                metadata["source"] = metadata["source"] + f"#_{i}"
                            document.metadata = metadata
                            yield document
                        return

                    if self.file_loader_cls or file["mimeType"] in self.conv_mapping:
                        logger.debug(
                            f"Try to convert '{file['name']}' with type "
                            f"'{file.get('mimeType', 'unknown')}'"
                        )
                        cls = self.conv_mapping[file["mimeType"]]
                        try:
                            documents = cls(file_path=str(path)).load()
                            for i, document in enumerate(documents):
                                metadata = self._extract_meta_data(file)
                                if "source" in metadata:
                                    metadata["source"] = metadata["source"] + f"#_{i}"
                                document.metadata = metadata
                                yield document
                            return
                        except Exception as e:
                            logger.warning(
                                f"Exception during the conversion of file "
                                f"'{file['name']}' ({e})"
                            )
                            return
                    else:
                        logger.warning(
                            f"Ignore '{file['name']}' with type "
                            f"'{file.get('mimeType', 'unknown')}'"
                        )
                        self._not_supported.add(file["mimeType"])
                    return
            except HttpError:
                logger.warning(
                    f"Impossible to convert the file '{file['name']}' ({file['id']})"
                )
                self._not_supported.add(file["mimeType"])
                return

    def _export_google_workspace_document(self, file: Dict) -> Iterator[Document]:
        if file["mimeType"] == "application/vnd.google-apps.document":
            return self._lazy_load_document_from_file(file)
        elif file["mimeType"] == "application/vnd.google-apps.spreadsheet":
            return self._lazy_load_sheet_from_file(file)
        elif file["mimeType"] == "application/vnd.google-apps.presentation":
            return self._lazy_load_slides_from_file(file)
        else:
            logger.warning(f" mimeType `{file['mimeType']}` not supported")
            return iter([])

    def _get_document(self, file: Dict, current_mode: str) -> Iterator[Document]:
        """Get text from file from Google Drive"""
        from googleapiclient.errors import HttpError

        mime_type = self._extract_mime_type(file)
        file["mimeType"] = mime_type

        # Manage shortcut
        if mime_type == "application/vnd.google-apps.shortcut":
            if not self.follow_shortcut:
                return
            if "shortcutDetails" not in file:
                logger.debug("Breaking shortcut without target_id")
                return
            target_id = file["shortcutDetails"]["targetId"]
            target_mime_type = file["shortcutDetails"]["targetMimeType"]
            description = file.get("description", "").strip()
            target_file = {
                "id": target_id,
                "mimeType": target_mime_type,
                "name": file["name"],
                "description": description,
            }
            # Search the description of the target_id
            target = self.files.get(
                fileId=target_id, supportsAllDrives=True, fields=self.fields
            ).execute()
            target_file["description"] = target.get(
                "description", target_file["description"]
            )
            if "webViewLink" in target:
                target_file["webViewLink"] = target["webViewLink"]
            if "webContentLink" in target:
                target_file["webContentLink"] = target["webContentLink"]
            logger.debug(f"Manage link {target_file}")
            if not current_mode.startswith("snippets"):
                documents = self._get_document(target_file, current_mode)
                for document in documents:
                    document.metadata["gdriveId"] = file[
                        "id"
                    ]  # Inject the id of the shortcut
                    yield document
                return
            else:
                if not description:
                    return iter([])
                yield Document(
                    page_content=description,
                    metadata={**self._extract_meta_data(target), **{"id": file["id"]}},
                )
        else:
            target_mime_type = mime_type

            # Fix document URL
            if target_mime_type not in [
                "application/vnd.google-apps.shortcut",
                "application/vnd.google-apps.folder",
            ]:
                logger.debug(
                    f"Manage file '{file['name']}' ({file['id']} - "
                    f"{file.get('mimeType')}) "
                )
            document_url = file.get(self.link_field)
            if not document_url:
                document_url = self._generate_missing_url(file)
            if not document_url:
                logger.debug(f"Impossible to find the url for file '{file['name']}")
            file[self.link_field] = document_url

            # if used only the description of the files to generate documents
            if current_mode.startswith("snippets"):
                if target_mime_type == "application/vnd.google-apps.folder":
                    self._folder_name_cache.put(file["id"], file["name"])
                    return

                if not self.filter(self, file):
                    logger.debug(f"Filter reject the file '{file['name']}")
                    return

                description = file.get("description", "").strip()
                if not description:  # Description with nothing
                    logger.debug(f"Empty description. Ignore file {file['name']}")
                    return

                logger.debug(
                    f"For file '{file['name']}' use the description '{description}'"
                )
                metadata = self._extract_meta_data(file)
                if "summary" in metadata:
                    del metadata["summary"]
                document = Document(page_content=description, metadata=metadata)
                logger.debug(f"Return '{document.page_content[:40]}...'")
                yield document
                return

            if target_mime_type == "application/vnd.google-apps.folder":
                self._folder_name_cache.put(file["id"], file["name"])
                return

            # Try to convert, download and extract text
            if target_mime_type.startswith("application/vnd.google-apps."):
                try:
                    if self.filter(self, file):
                        for doc in self._export_google_workspace_document(file):
                            yield doc
                    else:
                        logger.debug(f"Filter reject the document {file['name']}")
                        return
                except HttpError:
                    logger.warning(
                        f"Impossible to read or convert the content "
                        f"of '{file['name']}'' ({file['id']}"
                    )
                    return iter([])
            else:
                if self.filter(self, file):
                    try:
                        suffix = mimetypes.guess_extension(file["mimeType"])
                        if not suffix:
                            suffix = Path(file["name"]).suffix
                        if suffix not in self._not_supported:
                            for doc in self._lazy_load_file_from_file(file):
                                yield doc
                        else:
                            logger.debug(
                                f"Ignore mime-type '{file['mimeType']}' for file "
                                f"'{file['name']}'"
                            )
                    except HttpError as x:
                        logger.debug(
                            f"*** During recursive search, "
                            f"for file {file['name']}, ignore error {x}"
                        )
                else:
                    logger.debug(f"File '{file['mimeType']}' refused by the filter.")

    def _extract_meta_data(self, file: Dict) -> Dict:
        """
        Extract metadata from file

        :param file: The file
        :return: Dict the meta data
        """
        meta = {
            "gdriveId": file["id"],
            "mimeType": file["mimeType"],
            "name": file["name"],
            "title": file["name"],
        }
        if file[self.link_field]:
            meta["source"] = file[self.link_field]
        else:
            logger.debug(f"Invalid URL {file}")
        if "createdTime" in file:
            meta["createdTime"] = file["createdTime"]
        if "modifiedTime" in file:
            meta["modifiedTime"] = file["modifiedTime"]
        if "sha256Checksum" in file:
            meta["sha256Checksum"] = file["sha256Checksum"]
        if "owners" in file:
            meta["author"] = file["owners"][0]["displayName"]
        if file.get("description", "").strip():
            meta["summary"] = file["description"]
        return meta

    def lazy_get_relevant_documents(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> Iterator[Document]:
        """
        A generator to yield one document at a time.
        It's better for the memory.

        Args:
            query: Query string or None.
            kwargs: Additional parameters for templates of google list() api.

        Yield:
            Document
        """
        from googleapiclient.errors import HttpError

        from langchain import PromptTemplate as OriginalPromptTemplate

        if not query and "query" in self._kwargs:
            query = self._kwargs["query"]

        current_mode = kwargs.get("mode", self.mode)
        nb_yield = 0
        num_results = kwargs.get("num_results", self.num_results)
        if query is not None:
            # An empty query return all documents. But we want to return nothing.
            # We use a hack to replace the empty query to a random UUID.
            if not query:
                query = str(self.not_data)
            variables = {**self._kwargs, **kwargs, **{"query": query}}
        else:
            variables = {**self._kwargs, **kwargs}
        # Purge template variables
        variables = {
            k: v
            for k, v in variables.items()
            if k in cast(PromptTemplate, self._template).input_variables
        }
        query_str = (
            " "
            + "".join(cast(PromptTemplate, self._template).format(**variables))
            + " "
        )
        list_kwargs = {
            **self._gdrive_kwargs,
            **kwargs,
            **{
                "pageSize": max(100, int(num_results * 1.5))
                if num_results > 0
                else GoogleDriveUtilities._default_page_size,
                "fields": f"nextPageToken, files({self.fields})",
                "q": query_str,
            },
        }
        list_kwargs = {
            k: v for k, v in list_kwargs.items() if k in _acceptable_params_of_list
        }

        folder_id = variables.get("folder_id")
        documents_id: Set[str] = set()
        recursive_folders = []
        visited_folders = []
        try:
            while True:  # Manage current folder
                nextPageToken = None
                while True:  # Manage pages
                    list_kwargs["pageToken"] = nextPageToken
                    logger.debug(f"{query_str=}, {nextPageToken=}")
                    results = self.files.list(**list_kwargs).execute()
                    nextPageToken, files = (
                        results.get("nextPageToken"),
                        results["files"],
                    )
                    for file in files:
                        file_key = (
                            file.get("webViewLink")
                            or file.get("webContentLink")
                            or file["id"]
                        )
                        if file_key in file in documents_id:
                            logger.debug(f"Already yield the document {file['id']}")
                            continue
                        documents = self._get_document(file, current_mode)
                        for i, document in enumerate(documents):
                            document_key = (
                                document.metadata.get("source")
                                or document.metadata["gdriveId"]
                            )
                            if document_key in documents_id:
                                # May by, with a link
                                logger.debug(
                                    f"Already yield the document '{document_key}'"
                                )
                                continue
                            documents_id.add(document_key)
                            nb_yield += 1
                            logger.info(
                                f"Yield '{document.metadata['name']}'-{i} with "
                                f'"{GoogleDriveUtilities._snippet_from_page_content(document.page_content)}"'
                            )
                            yield document
                            if 0 < num_results == nb_yield:
                                break  # enough
                        if 0 < num_results == nb_yield:
                            break  # enough
                    if 0 < num_results == nb_yield:
                        break  # enough
                    if not nextPageToken:
                        break
                if not self.recursive:
                    break  # Not _recursive folder

                if 0 < num_results == nb_yield:
                    break  # enough

                # ----------- Search sub-directories
                if not re.search(r"'([^']*)'\s+in\s+parents", query_str):
                    break
                visited_folders.append(folder_id)
                try:
                    if not folder_id:
                        raise ValueError(
                            "Set 'folder_id' if you use 'recursive == True'"
                        )

                    nextPageToken = None
                    dir_template = OriginalPromptTemplate(
                        input_variables=["folder_id"],
                        template="(mimeType = 'application/vnd.google-apps.folder' "
                        "or mimeType = 'application/vnd.google-apps.shortcut') "
                        "and '{folder_id}' in parents and trashed=false",
                    )
                    subdir_query = "".join(dir_template.format(folder_id=folder_id))
                    while True:  # Manage pages
                        logger.debug(f"Search in subdir '{subdir_query}'")
                        list_kwargs = {
                            **self._gdrive_kwargs,
                            **kwargs,
                            **{
                                "pageSize": max(100, int(num_results * 1.5))
                                if num_results > 0
                                else GoogleDriveUtilities._default_page_size,
                                "fields": "nextPageToken, "
                                "files(id,name, mimeType, shortcutDetails)",
                            },
                        }
                        # Purge list_kwargs
                        list_kwargs = {
                            key: list_kwargs[key]
                            for key in list_kwargs
                            if key in GoogleDriveUtilities._gdrive_list_params
                        }
                        results = self.files.list(
                            pageToken=nextPageToken, q=subdir_query, **list_kwargs
                        ).execute()

                        nextPageToken, files = (
                            results.get("nextPageToken"),
                            results["files"],
                        )
                        for file in files:
                            try:
                                mime_type = GoogleDriveUtilities._extract_mime_type(
                                    file
                                )
                                if mime_type == "application/vnd.google-apps.folder":
                                    recursive_folders.append(file["id"])
                                    self._folder_name_cache.put(
                                        file["id"], file["name"]
                                    )
                                    logger.debug(
                                        f"Add the folder "
                                        f"'{file['name']}' ({file['id']})"
                                    )
                                if (
                                    mime_type == "application/vnd.google-apps.shortcut"
                                    and self.follow_shortcut
                                ):
                                    # Manage only shortcut to folder
                                    if "shortcutDetails" in file:
                                        target_mimetype = file["shortcutDetails"][
                                            "targetMimeType"
                                        ]
                                        if (
                                            target_mimetype
                                            == "application/vnd.google-apps.folder"
                                        ):
                                            recursive_folders.append(
                                                file["shortcutDetails"]["targetId"]
                                            )
                                    else:
                                        logger.debug(
                                            f"Breaking shortcut '{file['name']}' "
                                            f"('{file['id']}') to a folder."
                                        )
                            except HttpError as x:
                                # Error when manage recursive directory
                                logger.debug(
                                    f"*** During recursive search, ignore error {x}"
                                )

                        if not nextPageToken:
                            break

                    if not recursive_folders:
                        break

                    while True:
                        folder_id = recursive_folders.pop(0)
                        if folder_id not in visited_folders:
                            break
                    if not folder_id:
                        break

                    logger.debug(
                        f"Manage the folder '{self.get_folder_name(folder_id)}'"
                    )
                    # Update the parents folder and retry
                    query_str = re.sub(
                        r"'([^']*)'\s+in\s+parents",
                        f"'{folder_id}' in parents",
                        query_str,
                    )
                    list_kwargs["q"] = query_str
                except HttpError as x:
                    # Error when manage recursive directory
                    logger.debug(f"*** During recursive search, ignore error {x}")

        except HttpError as e:
            if "Invalid Value" in e.reason:
                raise
            logger.info(f"*** During google drive search, ignore error {e}")
            traceback.print_exc()

    def __del__(self) -> None:
        if hasattr(self, "_files") and self._files:
            self.files.close()
        if hasattr(self, "_docs") and self._docs:
            self._docs.close()
        if hasattr(self, "_spreadsheets") and self._spreadsheets:
            self._spreadsheets.close()
        if hasattr(self, "_slides") and self._slides:
            self._slides.close()

    @staticmethod
    def _extract_text(
        node: Any,
        *,
        key: str = "content",
        path: str = "/textRun",
        markdown: bool = True,
    ) -> List[str]:
        def visitor(result: List[str], node: Any, parent: str) -> List[str]:
            if isinstance(node, dict):
                if "paragraphMarker" in node:
                    result.append("- " if "bullet" in node["paragraphMarker"] else "")
                    return result
                if "paragraph" in node:
                    prefix=""
                    named_style_type= node['paragraph']["paragraphStyle"]["namedStyleType"]
                    level=re.match("HEADING_([1-9])",named_style_type)
                    if level:
                        prefix = f"{'#' * int(level[1])} "
                    if "bullet" in node["paragraph"]:
                        prefix +="- "
                    result.append(prefix)
                if "table" in node:
                    col_size = [0 for _ in range(node["table"]["columns"])]
                    rows = [[] for _ in range(node["table"]["rows"])]
                    for row_idx, row in enumerate(node["table"]["tableRows"]):
                        for col_idx, cell in enumerate(row["tableCells"]):
                            body = "".join(
                                visitor([], cell, parent + "/table/tableCells/[]")
                            )
                            # remove URL to calculate the col max size
                            pure_body = re.sub(r"\[(.*)\](?:\(.*\))", r"\1", body)
                            cell_size = max(3, len(max(pure_body.split("\n"), key=len)))
                            col_size[col_idx] = max(col_size[col_idx], cell_size)
                            str_cell = re.sub(r"\n", r"<br />", "".join(body).strip())
                            rows[row_idx].append(str_cell)
                    # Reformate to markdown with extra space
                    table_result = []
                    for row in rows:
                        for col_idx, cell in enumerate(row):
                            split_cell = re.split(r"(<br />|\n)", cell)
                            # Split each portion and pad with space
                            for i, portion in enumerate(split_cell):
                                if portion != "<br />":
                                    pure_portion = re.sub(
                                        r"\[(.*)\](?:\(.*\))", r"\1", portion
                                    )
                                    split_cell[i] = portion + (
                                        " " * (col_size[col_idx] - len(pure_portion))
                                    )
                            # rebuild the body
                            row[col_idx] = "".join(split_cell)
                    # Now, build a markdown array
                    for row_idx, row in enumerate(rows):
                        row_result = ["| "]
                        for cell in row:
                            row_result.append(f"{cell} | ")
                        result.append("".join(row_result) + "\n")
                        if row_idx == 0:
                            row_result = ["|"]
                            for col_idx in range(len(row)):
                                row_result.append(("-" * (col_size[col_idx] + 2)) + "|")
                            result.append("".join(row_result) + "\n")
                    return result

                if key in node and isinstance(node.get(key), str):
                    if parent.endswith(path):
                        if node[key].strip():
                            if markdown and (
                                ("style" in node and "link" in node["style"])
                                or ("textStyle" in node and "link" in node["textStyle"])
                            ):
                                style_node = (
                                    node["style"]
                                    if "style" in node
                                    else node["textStyle"]
                                )
                                link = style_node["link"]
                                if isinstance(link, dict):
                                    link = link["url"]
                                if link:
                                    result[-1] = f"{result[-1]}[{node[key]}]({link})"
                                else:
                                    # Empty link
                                    result[-1] = f"{result[-1]}{node[key]}"
                            else:
                                result[-1] = f"{result[-1]}{node[key]}"

                for k, v in node.items():
                    visitor(result, v, parent + "/" + k)
            elif isinstance(node, list):
                for v in node:
                    visitor(result, v, parent + "/[]")
            return result

        result = []
        visitor(result, node, "")
        # Clean the result:
        purge_result = []
        previous_empty = False
        for line in result:
            line = re.sub("\x0b\s*", "\n", line).strip()
            if not line:
                if previous_empty:
                    continue
                previous_empty = True
            else:
                previous_empty = False

            purge_result.append(line)
        return purge_result

    def _lazy_load_sheet_from_file(self, file: Dict) -> Iterator[Document]:
        """Load a sheet and all tabs from an ID."""

        if file["mimeType"] != "application/vnd.google-apps.spreadsheet":
            logger.warning(f"File with id '{file['id']}' is not a GSheet")
            return
        spreadsheet = self._spreadsheets.get(spreadsheetId=file["id"]).execute()
        sheets = spreadsheet.get("sheets", [])
        single: List[str] = []

        for sheet in sheets:
            sheet_name = sheet["properties"]["title"]
            result = (
                self._spreadsheets.values()
                .get(spreadsheetId=file["id"], range=sheet_name)
                .execute()
            )
            values = result.get("values", [])

            width = max([len(v) for v in values])
            headers = values[0]
            if self.gsheet_mode == "elements":
                for i, row in enumerate(values[1:], start=1):
                    content = []
                    for j, v in enumerate(row):
                        title = (
                            str(headers[j]).strip() + ": " if len(headers) > j else ""
                        )
                        content.append(f"{title}{str(v).strip()}")

                    raw_content = "\n".join(content)
                    metadata = self._extract_meta_data(file)
                    if "source" in metadata:
                        metadata["source"] = (
                            metadata["source"]
                            + "#gid="
                            + str(sheet["properties"]["sheetId"])
                            + f"&{i}"
                        )

                    yield Document(page_content=raw_content, metadata=metadata)
            elif self.gsheet_mode == "single":
                lines = []
                line = "|"
                i = 0
                for i, head in enumerate(headers):
                    line += head + "|"
                for _ in range(i, width - 1):
                    line += " |"

                lines.append(line)
                line = "|"
                for _ in range(width):
                    line += "---|"
                lines.append(line)
                for row in values[1:]:
                    line = "|"
                    for i, v in enumerate(row):
                        line += str(v).strip() + "|"
                    for _ in range(i, width - 1):
                        line += " |"

                    lines.append(line)
                raw_content = "\n".join(lines)
                single.append(raw_content)
                yield Document(
                    page_content="\n<PAGE BREAK>\n".join(single),
                    metadata=self._extract_meta_data(file),
                )
            else:
                raise ValueError(f"Invalid mode '{self.gslide_mode}'")

    def _only_obj(
        self,
        page_elements: Dict[str, Any],
        translateX: float = 0.0,
        translateY: float = 0.0,
    ):
        only_objets = []
        for obj in page_elements:
            if "elementGroup" in obj:
                group_translate_x = obj["transform"].get("translateX", 0)
                group_translate_y = obj["transform"].get("translateY", 0)
                only_objets.extend(
                    self._only_obj(
                        obj["elementGroup"]["children"],
                        translateX=group_translate_x,
                        translateY=group_translate_y,
                    )
                )
                pass
            elif "image" in obj:
                pass
            else:
                only_objets.append(obj)
        return only_objets

    def _sort_page_elements(
        self,
        page_elements: Dict[str, Any],
        translateX: float = 0.0,
        translateY: float = 0.0,
    ):
        only_obj = self._only_obj(page_elements, 0, 0)
        return sorted(
            only_obj,
            key=lambda x: (
                x["transform"].get("translateY", 0),
                x["transform"].get("translateX", 0),
            ),
        )

    def _lazy_load_slides_from_file(self, file: Dict) -> Iterator[Document]:
        """Load a GSlide. Split each slide to different documents"""
        if file["mimeType"] != "application/vnd.google-apps.presentation":
            logger.warning(f"File with id '{file['id']}' is not a GSlide")
            return
        gslide = self._slides.get(presentationId=file["id"]).execute()
        if self.gslide_mode == "single":
            lines = []
            for slide in gslide["slides"]:
                if "pageElements" in slide:
                    page_elements = self._sort_page_elements(slide["pageElements"])
                    lines.extend(GoogleDriveUtilities._extract_text(page_elements))
                    lines.append("<PAGE BREAK>")
            if lines:
                lines = lines[:-1]
            yield Document(
                page_content="\n".join(lines), metadata=self._extract_meta_data(file)
            )
        elif self.gslide_mode == "slide":
            for slide in gslide["slides"]:
                if "pageElements" in slide:
                    page_elements = self._sort_page_elements(slide["pageElements"])
                    meta = self._extract_meta_data(file).copy()
                    source = meta["source"]
                    if "#" in source:
                        source += f"&slide=id.{slide['objectId']}"
                    else:
                        source += f"#slide=id.{slide['objectId']}"
                    meta["source"] = source
                    yield Document(
                        page_content="\n".join(
                            GoogleDriveUtilities._extract_text(page_elements)
                        ),
                        metadata=meta,
                    )
        elif self.gslide_mode == "elements":
            for slide in gslide["slides"]:
                metadata = self._extract_meta_data(file)
                if "source" in metadata:
                    metadata["source"] = (
                        metadata["source"] + "#slide=file_id." + slide["objectId"]
                    )
                for slide in gslide["slides"]:
                    if "pageElements" in slide:
                        page_elements = self._sort_page_elements(slide["pageElements"])
                        for i, line in enumerate(
                            GoogleDriveUtilities._extract_text(page_elements)
                        ):
                            if line.strip():
                                m = metadata.copy()
                                if "source" in m:
                                    m["source"] = m["source"] + f"&i={i}"

                                yield Document(page_content=line, metadata=m)
        else:
            raise ValueError(f"Invalid gslide_mode '{self.gslide_mode}'")

    def _lazy_load_document_from_file(self, file: Dict) -> Iterator[Document]:
        """Load a GDocs."""
        if file["mimeType"] != "application/vnd.google-apps.document":
            logger.warning(f"File with id '{file['id']}' is not a GDoc")
        else:
            gdoc = self._docs.get(documentId=file["id"]).execute()
            text = GoogleDriveUtilities._extract_text(gdoc["body"]["content"])
            yield Document(
                page_content="\n".join(text), metadata=self._extract_meta_data(file)
            )

    def lazy_load_document_from_id(self, file_id: str) -> Iterator[Document]:
        return self._lazy_load_document_from_file(self._get_file_by_id(file_id=file_id))

    def load_document_from_id(self, file_id: str) -> List[Document]:
        """Load a GDocs."""
        from googleapiclient.errors import HttpError  # type: ignore

        try:
            return list(self.lazy_load_document_from_id(file_id))
        except HttpError:
            return []

    def load_slides_from_id(self, file_id: str) -> List[Document]:
        """Load a GSlide."""
        from googleapiclient.errors import HttpError  # type: ignore

        try:
            return list(
                self._lazy_load_slides_from_file(self._get_file_by_id(file_id=file_id))
            )
        except HttpError:
            return []

    def load_sheets_from_id(self, file_id: str) -> List[Document]:
        """Load a GSheets."""
        from googleapiclient.errors import HttpError  # type: ignore

        try:
            return list(
                self._lazy_load_sheet_from_file(self._get_file_by_id(file_id=file_id))
            )
        except HttpError:
            return []

    def lazy_load_file_from_id(self, file_id: str) -> Iterator[Document]:
        return self._get_document(
            self._get_file_by_id(file_id=file_id), current_mode=self.mode
        )

    def load_file_from_id(self, file_id: str) -> List[Document]:
        """Load file from GDrive"""
        from googleapiclient.errors import HttpError

        try:
            return list(self.lazy_load_file_from_id(file_id))
        except HttpError:
            return []


class GoogleDriveAPIWrapper(GoogleDriveUtilities):
    """
    Search on Google Drive.
    By default, search in filename only.
    Use a specific template if you want a different approach.
    """

    class Config:
        extra = Extra.allow
        underscore_attrs_are_private = True

    mode: Literal[
        "snippets", "snippets-markdown", "documents", "documents-markdown"
    ] = "snippets-markdown"

    num_results: int = 10
    """ Number of results """

    template: Union[
        PromptTemplate,
        Literal[
            "gdrive-all-in-folder",
            "gdrive-query",
            "gdrive-by-name",
            "gdrive-by-name-in-folder",
            "gdrive-query-in-folder",
            "gdrive-mime-type",
            "gdrive-mime-type-in-folder",
            "gdrive-query-with-mime-type",
            "gdrive-query-with-mime-type-and-folder",
        ],
        None,
    ] = None

    @root_validator(pre=True)
    def validate_template(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        folder_id = v.get("folder_id")

        if "template" not in v:
            if folder_id:
                template = get_template("gdrive-by-name-in-folder")
            else:
                template = get_template("gdrive-by-name")
            v["template"] = template
        return v

    def run(self, query: str) -> str:
        """Run query through Google Drive and parse result."""
        snippets = []
        logger.debug(f"{query=}")
        for document in self.lazy_get_relevant_documents(
            query=query, num_results=self.num_results
        ):
            content = document.page_content
            if (
                self.mode in ["snippets", "snippets-markdown"]
                and "summary" in document.metadata
                and document.metadata["summary"]
            ):
                content = document.metadata["summary"]
            if self.mode == "snippets":
                snippets.append(
                    f"Name: {document.metadata['name']}\n"
                    f"Source: {document.metadata['source']}\n" + f"Summary: {content}"
                )
            elif self.mode == "snippets-markdown":
                snippets.append(
                    f"[{document.metadata['name']}]"
                    f"({document.metadata['source']})<br/>\n" + f"{content}"
                )
            elif self.mode == "documents":
                snippets.append(
                    f"Name: {document.metadata['name']}\n"
                    f"Source: {document.metadata['source']}\n" + f"Summary: "
                    f"{GoogleDriveUtilities._snippet_from_page_content(content)}"
                )
            elif self.mode == "documents-markdown":
                snippets.append(
                    f"[{document.metadata['name']}]"
                    f"({document.metadata['source']})<br/>"
                    + f"{GoogleDriveUtilities._snippet_from_page_content(content)}"
                )
            else:
                raise ValueError(f"Invalid mode `{self.mode}`")

        if not len(snippets):
            return "No document found"

        return "\n\n".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through Google Drive and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            Like bing_search, a list of dictionaries with the following keys:
                `snippet: The `description` of the result.
                `title`: The title of the result.
                `link`: The link to the result.
        """
        metadata_results = []
        for document in self.lazy_get_relevant_documents(
            query=query, num_results=num_results
        ):
            metadata_result = {
                "title": document.metadata["name"],
                "link": document.metadata["source"],
            }
            if "summary" in document.metadata:
                metadata_result["snippet"] = document.metadata["summary"]
            else:
                metadata_result[
                    "snippet"
                ] = GoogleDriveAPIWrapper._snippet_from_page_content(
                    document.page_content
                )
            metadata_results.append(metadata_result)
        if not metadata_results:
            return [{"Result": "No good Google Drive Search Result was found"}]

        return metadata_results

    def get_format_instructions(self) -> str:
        """Return format instruction for LLM"""
        return FORMAT_INSTRUCTION
