"""Util that calls Box APIs."""

from enum import Enum
from typing import Any, Dict, List, Optional

import box_sdk_gen  # type: ignore
import requests
from langchain_core.documents import Document
from langchain_core.utils import from_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class DocumentFiles(Enum):
    """DocumentFiles(Enum).

    An enum containing all of the supported extensions for files
    Box considers Documents. These files should have text
    representations.
    """

    DOC = "doc"
    DOCX = "docx"
    GDOC = "gdoc"
    GSHEET = "gsheet"
    NUMBERS = "numbers"
    ODS = "ods"
    ODT = "odt"
    PAGES = "pages"
    PDF = "pdf"
    RTF = "rtf"
    WPD = "wpd"
    XLS = "xls"
    XLSM = "xlsm"
    XLSX = "xlsx"
    AS = "as"
    AS3 = "as3"
    ASM = "asm"
    BAT = "bat"
    C = "c"
    CC = "cc"
    CMAKE = "cmake"
    CPP = "cpp"
    CS = "cs"
    CSS = "css"
    CSV = "csv"
    CXX = "cxx"
    DIFF = "diff"
    ERB = "erb"
    GROOVY = "groovy"
    H = "h"
    HAML = "haml"
    HH = "hh"
    HTM = "htm"
    HTML = "html"
    JAVA = "java"
    JS = "js"
    JSON = "json"
    LESS = "less"
    LOG = "log"
    M = "m"
    MAKE = "make"
    MD = "md"
    ML = "ml"
    MM = "mm"
    MSG = "msg"
    PHP = "php"
    PL = "pl"
    PROPERTIES = "properties"
    PY = "py"
    RB = "rb"
    RST = "rst"
    SASS = "sass"
    SCALA = "scala"
    SCM = "scm"
    SCRIPT = "script"
    SH = "sh"
    SML = "sml"
    SQL = "sql"
    TXT = "txt"
    VI = "vi"
    VIM = "vim"
    WEBDOC = "webdoc"
    XHTML = "xhtml"
    XLSB = "xlsb"
    XML = "xml"
    XSD = "xsd"
    XSL = "xsl"
    YAML = "yaml"
    GSLLIDE = "gslide"
    GSLIDES = "gslides"
    KEY = "key"
    ODP = "odp"
    PPT = "ppt"
    PPTX = "pptx"
    BOXNOTE = "boxnote"


class ImageFiles(Enum):
    """ImageFiles(Enum).

    An enum containing all of the supported extensions for files
    Box considers images.
    """

    ARW = "arw"
    BMP = "bmp"
    CR2 = "cr2"
    DCM = "dcm"
    DICM = "dicm"
    DICOM = "dicom"
    DNG = "dng"
    EPS = "eps"
    EXR = "exr"
    GIF = "gif"
    HEIC = "heic"
    INDD = "indd"
    INDML = "indml"
    INDT = "indt"
    INX = "inx"
    JPEG = "jpeg"
    JPG = "jpg"
    NEF = "nef"
    PNG = "png"
    SVG = "svg"
    TIF = "tif"
    TIFF = "tiff"
    TGA = "tga"
    SVS = "svs"


class BoxAuthType(Enum):
    """BoxAuthType(Enum).

    an enum to tell BoxLoader how you wish to autheticate your Box connection.

    Options are:

    TOKEN - Use a developer token generated from the Box Deevloper Token.
            Only recommended for development.
            Provide ``box_developer_token``.
    CCG - Client Credentials Grant.
          provide ``box_client_id`, ``box_client_secret`,
          and ``box_enterprise_id`` or optionally `box_user_id`.
    JWT - Use JWT for authentication. Config should be stored on the file
          system accessible to your app.
          provide ``box_jwt_path``. Optionally, provide ``box_user_id`` to
          act as a specific user
    """

    TOKEN = "token"
    """Use a developer token or a token retrieved from ``box-sdk-gen``"""

    CCG = "ccg"
    """Use ``client_credentials`` type grant"""

    JWT = "jwt"
    """Use JWT bearer token auth"""


class BoxAuth(BaseModel):
    """**BoxAuth.**

    The ``box-langchain`` package offers some flexibility to authentication. The
    most basic authentication method is by using a developer token. This can be
    found in the `Box developer console <https://account.box.com/developers/console>`_
    on the configuration screen. This token is purposely short-lived (1 hour) and is
    intended for development. With this token, you can add it to your environment as
    ``BOX_DEVELOPER_TOKEN``, you can pass it directly to the loader, or you can use the
    ``BoxAuth`` authentication helper class.

    `BoxAuth` supports the following authentication methods:

    * **Token** — either a developer token or any token generated through the Box SDK
    * **JWT** with a service account
    * **JWT** with a specified user
    * **CCG** with a service account
    * **CCG** with a specified user

    .. note::
        If using JWT authentication, you will need to download the configuration from
        the Box developer console after generating your public/private key pair. Place
        this file in your application directory structure somewhere. You will use the
        path to this file when using the ``BoxAuth`` helper class. If you wish to use
        OAuth2 with the authorization_code flow, please use ``BoxAuthType.TOKEN`` with
        the token you have acquired.

    For more information, learn about how to
    `set up a Box application <https://developer.box.com/guides/getting-started/first-application/>`_,
    and check out the
    `Box authentication guide <https://developer.box.com/guides/authentication/select/>`_
    for more about our different authentication options.

    Simple implementation:

        To instantiate, you must provide a ``langchain_box.utilities.BoxAuthType``.

        BoxAuthType is an enum to tell BoxLoader how you wish to autheticate your
        Box connection.

        Options are:

        TOKEN - Use a developer token generated from the Box Deevloper Token.
                Only recommended for development.
                Provide ``box_developer_token``.
        CCG - Client Credentials Grant.
            provide ``box_client_id``, ``box_client_secret``,
            and ``box_enterprise_id`` or optionally ``box_user_id``.
        JWT - Use JWT for authentication. Config should be stored on the file
            system accessible to your app.
            provide ``box_jwt_path``. Optionally, provide ``box_user_id`` to
            act as a specific user

        **Examples**:

        **Token**

        .. code-block:: python

            from langchain_box.document_loaders import BoxLoader
            from langchain_box.utilities import BoxAuth, BoxAuthType

            auth = BoxAuth(
                auth_type=BoxAuthType.TOKEN,
                box_developer_token=box_developer_token
            )

            loader = BoxLoader(
                box_auth=auth,
                ...
            )


        **JWT with a service account**

        .. code-block:: python

            from langchain_box.document_loaders import BoxLoader
            from langchain_box.utilities import BoxAuth, BoxAuthType

            auth = BoxAuth(
                auth_type=BoxAuthType.JWT,
                box_jwt_path=box_jwt_path
            )

            loader = BoxLoader(
                box_auth=auth,
                ...
            )


        **JWT with a specified user**

        .. code-block:: python

            from langchain_box.document_loaders import BoxLoader
            from langchain_box.utilities import BoxAuth, BoxAuthType

            auth = BoxAuth(
                auth_type=BoxAuthType.JWT,
                box_jwt_path=box_jwt_path,
                box_user_id=box_user_id
            )

            loader = BoxLoader(
                box_auth=auth,
                ...
            )


        **CCG with a service account**

        .. code-block:: python

            from langchain_box.document_loaders import BoxLoader
            from langchain_box.utilities import BoxAuth, BoxAuthType

            auth = BoxAuth(
                auth_type=BoxAuthType.CCG,
                box_client_id=box_client_id,
                box_client_secret=box_client_secret,
                box_enterprise_id=box_enterprise_id
            )

            loader = BoxLoader(
                box_auth=auth,
                ...
            )


        **CCG with a specified user**

        .. code-block:: python

            from langchain_box.document_loaders import BoxLoader
            from langchain_box.utilities import BoxAuth, BoxAuthType

            auth = BoxAuth(
                auth_type=BoxAuthType.CCG,
                box_client_id=box_client_id,
                box_client_secret=box_client_secret,
                box_user_id=box_user_id
            )

            loader = BoxLoader(
                box_auth=auth,
                ...
            )

    """

    auth_type: BoxAuthType
    """``langchain_box.utilities.BoxAuthType``. Enum describing how to
       authenticate against Box"""

    box_developer_token: Optional[str] = Field(
        default_factory=from_env("BOX_DEVELOPER_TOKEN", default=None)
    )
    """ If using ``BoxAuthType.TOKEN``, provide your token here"""

    box_jwt_path: Optional[str] = Field(
        default_factory=from_env("BOX_JWT_PATH", default=None)
    )
    """If using ``BoxAuthType.JWT``, provide local path to your
       JWT configuration file"""

    box_client_id: Optional[str] = Field(
        default_factory=from_env("BOX_CLIENT_ID", default=None)
    )
    """If using ``BoxAuthType.CCG``, provide your app's client ID"""

    box_client_secret: Optional[str] = Field(
        default_factory=from_env("BOX_CLIENT_SECRET", default=None)
    )
    """If using ``BoxAuthType.CCG``, provide your app's client secret"""

    box_enterprise_id: Optional[str] = None
    """If using ``BoxAuthType.CCG``, provide your enterprise ID.
       Only required if you are not sending ``box_user_id``"""

    box_user_id: Optional[str] = None
    """If using ``BoxAuthType.CCG`` or ``BoxAuthType.JWT``, providing 
       ``box_user_id`` will act on behalf of a specific user"""

    _box_client: Optional[box_sdk_gen.BoxClient] = None
    _custom_header: Dict = dict({"x-box-ai-library": "langchain"})

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def validate_box_auth_inputs(self) -> Self:
        """Validate auth_type is set"""
        if not self.auth_type:
            raise ValueError("Auth type must be set.")

        """Validate that TOKEN auth type provides box_developer_token."""
        if self.auth_type == "token" and not self.box_developer_token:
            raise ValueError(f"{self.auth_type} requires box_developer_token to be set")

        """Validate that JWT auth type provides box_jwt_path."""
        if self.auth_type == "jwt" and not self.box_jwt_path:
            raise ValueError(f"{self.auth_type} requires box_jwt_path to be set")

        """Validate that CCG auth type provides box_client_id and
           box_client_secret and either box_enterprise_id or box_user_id."""
        if self.auth_type == "ccg":
            if (
                not self.box_client_id
                or not self.box_client_secret
                or (not self.box_enterprise_id and not self.box_user_id)
            ):
                raise ValueError(
                    f"{self.auth_type} requires box_client_id, \
                        box_client_secret, and box_enterprise_id/box_user_id."
                )

        return self

    def _authorize(self) -> None:
        if self.auth_type == "token":
            try:
                auth = box_sdk_gen.BoxDeveloperTokenAuth(token=self.box_developer_token)
                self._box_client = box_sdk_gen.BoxClient(auth=auth).with_extra_headers(
                    extra_headers=self._custom_header
                )

            except box_sdk_gen.BoxSDKError as bse:
                raise RuntimeError(
                    f"Error getting client from developer token: {bse.message}"
                )
            except Exception as ex:
                raise ValueError(
                    f"Invalid Box developer token. Please verify your \
                        token and try again.\n{ex}"
                ) from ex

        elif self.auth_type == "jwt":
            try:
                jwt_config = box_sdk_gen.JWTConfig.from_config_file(
                    config_file_path=self.box_jwt_path
                )
                auth = box_sdk_gen.BoxJWTAuth(config=jwt_config)

                self._box_client = box_sdk_gen.BoxClient(auth=auth).with_extra_headers(
                    extra_headers=self._custom_header
                )

                if self.box_user_id is not None:
                    user_auth = auth.with_user_subject(self.box_user_id)
                    self._box_client = box_sdk_gen.BoxClient(
                        auth=user_auth
                    ).with_extra_headers(extra_headers=self._custom_header)

            except box_sdk_gen.BoxSDKError as bse:
                raise RuntimeError(
                    f"Error getting client from jwt token: {bse.message}"
                )
            except Exception as ex:
                raise ValueError(
                    "Error authenticating. Please verify your JWT config \
                        and try again."
                ) from ex

        elif self.auth_type == "ccg":
            try:
                if self.box_user_id is not None:
                    ccg_config = box_sdk_gen.CCGConfig(
                        client_id=self.box_client_id,
                        client_secret=self.box_client_secret,
                        user_id=self.box_user_id,
                    )
                else:
                    ccg_config = box_sdk_gen.CCGConfig(
                        client_id=self.box_client_id,
                        client_secret=self.box_client_secret,
                        enterprise_id=self.box_enterprise_id,
                    )
                auth = box_sdk_gen.BoxCCGAuth(config=ccg_config)

                self._box_client = box_sdk_gen.BoxClient(auth=auth).with_extra_headers(
                    extra_headers=self._custom_header
                )

            except box_sdk_gen.BoxSDKError as bse:
                raise RuntimeError(
                    f"Error getting client from ccg token: {bse.message}"
                )
            except Exception as ex:
                raise ValueError(
                    "Error authenticating. Please verify you are providing a \
                        valid client id, secret and either a valid user ID or \
                            enterprise ID."
                ) from ex

        else:
            raise ValueError(
                f"{self.auth_type} is not a valid auth_type. Value must be \
            TOKEN, CCG, or JWT."
            )

    def get_client(self) -> box_sdk_gen.BoxClient:
        """Instantiate the Box SDK."""
        if self._box_client is None:
            self._authorize()

        return self._box_client


class SearchTypeFilter(Enum):
    """SearchTypeFilter.

    Enum to limit the what we search.
    """

    NAME = "name"
    """The name of the item, as defined by its ``name`` field."""

    DESCRIPTION = "description"
    """The description of the item, as defined by its ``description`` field."""

    FILE_CONTENT = "file_content"
    """The actual content of the file."""

    COMMENTS = "comments"
    """The content of any of the comments on a file or folder."""

    TAGS = "tags"
    """Any tags that are applied to an item, as defined by its ``tags`` field."""


class BoxSearchOptions(BaseModel):
    ancestor_folder_ids: Optional[List[str]] = None
    """Limits the search results to items within the given list of folders, 
       defined as a comma separated lists of folder IDs."""

    search_type_filter: Optional[List[SearchTypeFilter]] = None
    """Limits the search results to any items that match the search query for a 
       specific part of the file, for example the file description.

       Content types are defined as a comma separated lists of Box recognized 
       content types. The allowed content types are as follows. Default is all."""

    created_date_range: Optional[List[str]] = None
    """Limits the search results to any items created within a given date range.

       Date ranges are defined as comma separated RFC3339 timestamps.

       If the the start date is omitted (,2014-05-17T13:35:01-07:00) anything 
       created before the end date will be returned.

       If the end date is omitted (2014-05-15T13:35:01-07:00,) the current 
       date will be used as the end date instead."""

    file_extensions: Optional[List[DocumentFiles]] = None
    """Limits the search results to any files that match any of the provided 
        file extensions. This list is a comma-separated list of 
        ``langchain_box.utilities.DocumentFiles`` entries"""

    k: Optional[int] = 100
    """Defines the maximum number of items to return. Defaults to 100, maximum 
        is 200."""

    size_range: Optional[List[int]] = None
    """Limits the search results to any items with a size within a given file 
        size range. This applied to files and folders.

        Size ranges are defined as comma separated list of a lower and upper 
        byte size limit (inclusive).

        The upper and lower bound can be omitted to create open ranges."""

    updated_date_range: Optional[List[str]] = None
    """Limits the search results to any items updated within a given date range.

        Date ranges are defined as comma separated RFC3339 timestamps.

        If the start date is omitted (,2014-05-17T13:35:01-07:00) anything 
        updated before the end date will be returned.

        If the end date is omitted (2014-05-15T13:35:01-07:00,) the current 
        date will be used as the end date instead."""

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = "allow"

    @model_validator(mode="after")
    def validate_search_options(self) -> Self:
        """Validate k is between 1 and 200"""
        if self.k > 200 or self.k < 1:  # type: ignore[operator]
            raise ValueError(
                f"Invalid setting of k {self.k}. " "Value must be between 1 and 200."
            )

        """Validate created_date_range start date is before end date"""
        if self.created_date_range:
            if (
                self.created_date_range[0] is None  # type: ignore[index]
                or self.created_date_range[0] == ""  # type: ignore[index]
                or self.created_date_range[1] is None  # type: ignore[index]
                or self.created_date_range[1] == ""  # type: ignore[index]
            ):
                pass
            else:
                if (
                    self.created_date_range[0]  # type: ignore[index]
                    > self.created_date_range[1]  # type: ignore[index]
                ):
                    raise ValueError("Start date must be before end date.")

        """Validate updated_date_range start date is before end date"""
        if self.updated_date_range:
            if (
                self.updated_date_range[0] is None  # type: ignore[index]
                or self.updated_date_range[0] == ""  # type: ignore[index]
                or self.updated_date_range[1] is None  # type: ignore[index]
                or self.updated_date_range[1] == ""  # type: ignore[index]
            ):
                pass
            else:
                if (
                    self.updated_date_range[0]  # type: ignore[index]
                    > self.updated_date_range[1]  # type: ignore[index]
                ):
                    raise ValueError("Start date must be before end date.")

        return self


class _BoxAPIWrapper(BaseModel):
    """Wrapper for Box API."""

    box_developer_token: Optional[str] = Field(
        default_factory=from_env("BOX_DEVELOPER_TOKEN", default=None)
    )
    """String containing the Box Developer Token generated in the developer console"""

    box_auth: Optional[BoxAuth] = None
    """Configured langchain_box.utilities.BoxAuth object"""

    character_limit: Optional[int] = -1
    """character_limit is an int that caps the number of characters to
       return per document."""

    box_search_options: Optional[BoxSearchOptions] = None
    """Search options to configure BoxRetriever to narrow search results."""

    _box: Optional[box_sdk_gen.BoxClient]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        extra="allow",
    )

    @model_validator(mode="after")
    def validate_box_api_inputs(self) -> Self:
        self._box = None

        """Validate that TOKEN auth type provides box_developer_token."""
        if not self.box_auth:
            if not self.box_developer_token:
                raise ValueError(
                    "You must configure either box_developer_token of box_auth"
                )
        else:
            box_auth = self.box_auth
            self._box = box_auth.get_client()  #  type: ignore[union-attr]

        return self

    def get_box_client(self) -> box_sdk_gen.BoxClient:
        box_auth = BoxAuth(
            auth_type=BoxAuthType.TOKEN, box_developer_token=self.box_developer_token
        )

        self._box = box_auth.get_client()

    def _do_request(self, url: str) -> Any:
        try:
            access_token = self._box.auth.retrieve_token().access_token  #  type: ignore[union-attr]
        except box_sdk_gen.BoxSDKError as bse:
            raise RuntimeError(f"Error getting client from jwt token: {bse.message}")

        resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
        resp.raise_for_status()
        return resp.content

    def _get_text_representation(self, file_id: str = "") -> tuple[str, str, str]:
        try:
            from box_sdk_gen import BoxAPIError, BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")

        if self._box is None:
            self.get_box_client()

        try:
            file = self._box.files.get_file_by_id(  #  type: ignore[union-attr]
                file_id,
                x_rep_hints="[extracted_text]",
                fields=["name", "representations", "type"],
            )
        except BoxAPIError as bae:
            raise RuntimeError(f"BoxAPIError: Error getting text rep: {bae.message}")
        except BoxSDKError as bse:
            raise RuntimeError(f"BoxSDKError: Error getting text rep: {bse.message}")
        except Exception:
            return None, None, None  #   type: ignore[return-value]

        file_repr = file.representations.entries

        if len(file_repr) <= 0:
            return None, None, None  #   type: ignore[return-value]

        for entry in file_repr:
            if entry.representation == "extracted_text":
                # If the file representation doesn't exist, calling
                # info.url will generate text if possible
                if entry.status.state == "none":
                    self._do_request(entry.info.url)

                url = entry.content.url_template.replace("{+asset_path}", "")
                file_name = file.name.replace(".", "_").replace(" ", "_")

                try:
                    raw_content = self._do_request(url)
                except requests.exceptions.HTTPError:
                    return None, None, None  #   type: ignore[return-value]

                if (
                    self.character_limit is not None and self.character_limit > 0  #   type: ignore[operator]
                ):
                    content = raw_content[0 : (self.character_limit - 1)]
                else:
                    content = raw_content

                return file_name, content, url

        return None, None, None  #   type: ignore[return-value]

    def get_document_by_file_id(self, file_id: str) -> Optional[Document]:
        """Load a file from a Box id. Accepts file_id as str.
        Returns `Document`"""

        if self._box is None:
            self.get_box_client()

        file = self._box.files.get_file_by_id(  #  type: ignore[union-attr]
            file_id, fields=["name", "type", "extension"]
        )

        if file.type == "file":
            if hasattr(DocumentFiles, file.extension.upper()):
                file_name, content, url = self._get_text_representation(file_id=file_id)

                if file_name is None or content is None or url is None:
                    return None

                metadata = {
                    "source": f"{url}",
                    "title": f"{file_name}",
                }

                return Document(page_content=content, metadata=metadata)

            return None

        return None

    def get_folder_items(self, folder_id: str) -> box_sdk_gen.Items:
        """Get all the items in a folder. Accepts folder_id as str.
        returns box_sdk_gen.Items"""
        if self._box is None:
            self.get_box_client()

        try:
            folder_contents = self._box.folders.get_folder_items(  #  type: ignore[union-attr]
                folder_id, fields=["id", "type", "name"]
            )
        except box_sdk_gen.BoxAPIError as bae:
            raise RuntimeError(
                f"BoxAPIError: Error getting folder content: {bae.message}"
            )
        except box_sdk_gen.BoxSDKError as bse:
            raise RuntimeError(
                f"BoxSDKError: Error getting folder content: {bse.message}"
            )

        return folder_contents.entries

    def search_box(self, query: str) -> List[Document]:
        if self._box is None:
            self.get_box_client()

        files = []

        try:
            results = None

            if self.box_search_options is None:
                results = self._box.search.search_for_content(  #  type: ignore[union-attr]
                    query=query, fields=["id", "type", "extension"], type="file"
                )
            else:
                results = self._box.search.search_for_content(  #  type: ignore[union-attr]
                    query=query,
                    fields=["id", "type", "extension"],
                    type="file",
                    ancestor_folder_ids=self.box_search_options.ancestor_folder_ids,  #  type: ignore[union-attr]
                    content_types=self.box_search_options.search_type_filter,  #  type: ignore[union-attr]
                    created_at_range=self.box_search_options.created_date_range,  #  type: ignore[union-attr]
                    file_extensions=self.box_search_options.file_extensions,  #  type: ignore[union-attr]
                    limit=self.box_search_options.k,  #  type: ignore[union-attr]
                    size_range=self.box_search_options.size_range,  #  type: ignore[union-attr]
                    updated_at_range=self.box_search_options.updated_date_range,  #  type: ignore[union-attr]
                )

            if results.entries is None or len(results.entries) <= 0:
                return None  #  type: ignore[return-value]

            for file in results.entries:
                if (
                    file is not None
                    and file.type == "file"
                    and hasattr(DocumentFiles, file.extension.upper())
                ):
                    doc = self.get_document_by_file_id(file.id)

                    if doc is not None:
                        files.append(doc)

            return files
        except box_sdk_gen.BoxAPIError as bae:
            raise RuntimeError(
                f"BoxAPIError: Error getting search results: {bae.message}"
            )
        except box_sdk_gen.BoxSDKError as bse:
            raise RuntimeError(
                f"BoxSDKError: Error getting search results: {bse.message}"
            )

    def ask_box_ai(
        self,
        query: str,
        box_file_ids: List[str],
        answer: bool = True,
        citations: bool = False,
    ) -> List[Document]:
        if self._box is None:
            self.get_box_client()

        ai_mode = box_sdk_gen.CreateAiAskMode.SINGLE_ITEM_QA.value

        if len(box_file_ids) > 1:
            ai_mode = box_sdk_gen.CreateAiAskMode.MULTIPLE_ITEM_QA.value
        elif len(box_file_ids) <= 0:
            raise ValueError("BOX_AI_ASK requires at least one file ID")

        items = []

        for file_id in box_file_ids:
            item = box_sdk_gen.AiItemBase(
                id=file_id, type=box_sdk_gen.AiItemBaseTypeField.FILE.value
            )
            items.append(item)

        try:
            response = self._box.ai.create_ai_ask(  #  type: ignore[union-attr]
                mode=ai_mode, prompt=query, items=items, include_citations=citations
            )

        except box_sdk_gen.BoxAPIError as bae:
            raise RuntimeError(
                f"BoxAPIError: Error getting Box AI result: {bae.message}"
            )
        except box_sdk_gen.BoxSDKError as bse:
            raise RuntimeError(
                f"BoxSDKError: Error getting Box AI result: {bse.message}"
            )

        docs = []

        if answer:
            content = response.answer
            metadata = {"source": "Box AI", "title": f"Box AI {query}"}

            document = Document(page_content=content, metadata=metadata)
            docs.append(document)

        if citations:
            box_citations = response.citations

            for citation in box_citations:
                content = citation.content
                file_name = citation.name
                file_id = citation.id
                file_type = citation.type.value

                metadata = {
                    "source": f"Box AI {query}",
                    "file_name": file_name,
                    "file_id": file_id,
                    "file_type": file_type,
                }

                document = Document(page_content=content, metadata=metadata)
                docs.append(document)

        return docs
