"""Util that calls Box APIs."""

from enum import Enum
from typing import Any, Dict, Optional

import box_sdk_gen  # type: ignore
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env


class DocumentFiles(Enum):
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


"""
    BoxAuthType 
    an enum to tell BoxLoader how you wish to autheticate your Box connection.

    Options are:
    TOKEN - Use a developer token generated from the Box Deevloper Token.
            Only recommended for development.
            Provide `box_developer_token`.
    CCG - Client Credentials Grant.
          provide `box_client_id`, `box_client_secret`,
          and `box_enterprise_id` or optionally `box_user_id`.
    JWT - Use JWT for authentication. Config should be stored on the file
          system accessible to your app.
          provide `box_jwt_path`. Optionally, provide `box_user_id` to 
          act as a specific user
"""


class BoxAuthType(Enum):
    """Use a developer token or a token retrieved from box-sdk-gen"""

    TOKEN = "token"
    """Use `client_credentials` type grant"""
    CCG = "ccg"
    """Use JWT bearer token auth"""
    JWT = "jwt"


"""
    `BoxAuth` supports the following authentication methods:

    * Token — either a developer token or any token generated through the Box SDK
    * JWT with a service account
    * JWT with a specified user
    * CCG with a service account
    * CCG with a specified user

    :::note
    If using JWT authentication, you will need to download the configuration from the 
    Box developer console after generating your public/private key pair. Place this  
    file in your application directory structure somewhere. You will use the path to  
    this file when using the `BoxAuth` helper class.
    :::

    For more information, learn about how to 
    [set up a Box application](https://developer.box.com/guides/getting-started/first-application/),
    and check out the 
    [Box authentication guide](https://developer.box.com/guides/authentication/select/)
    for more about our different authentication options.

    Simple implementation

        To instantiate, you must provide a ``langchain_box.utilities.BoxAuthType``.

        BoxAuthType is an enum to tell BoxLoader how you wish to autheticate your 
        Box connection.

        Options are:
        TOKEN - Use a developer token generated from the Box Deevloper Token.
                Only recommended for development.
                Provide `box_developer_token`.
        CCG - Client Credentials Grant.
            provide `box_client_id`, `box_client_secret`,
            and `box_enterprise_id` or optionally `box_user_id`.
        JWT - Use JWT for authentication. Config should be stored on the file
            system accessible to your app.
            provide `box_jwt_path`. Optionally, provide `box_user_id` to 
            act as a specific user

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

    To see examples for each supported authentication methodology, visit the 
    [Box providers](/docs/integrations/providers/box) page. If you want to 
    use OAuth 2.0 `authorization_code` flow, use 
    [box-sdk-gen](https://github.com/box/box-python-sdk-gen) SDK, get your 
    token, and use `BoxAuthType.TOKEN` type.
"""


class BoxAuth(BaseModel):
    """Authentication type to use. Must pass BoxAuthType enum"""

    auth_type: BoxAuthType
    """ If using BoxAuthType.TOKEN, provide your token here"""
    box_developer_token: Optional[str] = None
    """If using BoxAuthType.JWT, provide local path to your
       JWT configuration file"""
    box_jwt_path: Optional[str] = None
    """If using BoxAuthType.CCG, provide your app's client ID"""
    box_client_id: Optional[str] = None
    """If using BoxAuthType.CCG, provide your app's client secret"""
    box_client_secret: Optional[str] = None
    """If using BoxAuthType.CCG, provide your enterprise ID.
       Only required if you are not sending `box_user_id`"""
    box_enterprise_id: Optional[str] = None
    """If using BoxAuthType.CCG or BoxAuthType.JWT, providing 
       `box_user_id` will act on behalf of a specific user"""
    box_user_id: Optional[str] = None

    box_client: Optional[box_sdk_gen.BoxClient] = None
    custom_header: Dict = dict({"x-box-ai-library": "langchain"})

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = "allow"

    @root_validator()
    def validate_box_auth_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate auth_type is set"""
        if not values.get("auth_type"):
            raise ValueError("Auth type must be set.")

        """Validate that TOKEN auth type provides box_developer_token."""
        if values.get("auth_type") == "token":
            if not get_from_dict_or_env(
                values, "box_developer_token", "BOX_DEVELOPER_TOKEN"
            ):
                raise ValueError(
                    f"{values.get('auth_type')} requires box_developer_token to be set"
                )

        """Validate that JWT auth type provides box_jwt_path."""
        if values.get("auth_type") == "jwt":
            if not get_from_dict_or_env(values, "box_jwt_path", "BOX_JWT_PATH"):
                raise ValueError(
                    f"{values.get('auth_type')} requires box_jwt_path to be set"
                )

        """Validate that CCG auth type provides box_client_id and
           box_client_secret and either box_enterprise_id or box_user_id."""
        if values.get("auth_type") == "ccg":
            if (
                not get_from_dict_or_env(values, "box_client_id", "BOX_CLIENT_ID")
                or not get_from_dict_or_env(
                    values, "box_client_secret", "BOX_CLIENT_SECRET"
                )
                or (
                    not values.get("box_enterprise_id")
                    and not values.get("box_user_id")
                )
            ):
                raise ValueError(
                    f"{values.get('auth_type')} requires box_client_id, \
                        box_client_secret, and box_enterprise_id."
                )

        return values

    def authorize(self) -> None:
        match self.auth_type:
            case "token":
                try:
                    auth = box_sdk_gen.BoxDeveloperTokenAuth(
                        token=self.box_developer_token
                    )
                    self.box_client = box_sdk_gen.BoxClient(
                        auth=auth
                    ).with_extra_headers(extra_headers=self.custom_header)

                except box_sdk_gen.BoxSDKError as bse:
                    raise RuntimeError(
                        f"Error getting client from developer token: {bse.message}"
                    )
                except Exception as ex:
                    raise ValueError(
                        f"Invalid Box developer token. Please verify your \
                            token and try again.\n{ex}"
                    ) from ex

            case "jwt":
                try:
                    jwt_config = box_sdk_gen.JWTConfig.from_config_file(
                        config_file_path=self.box_jwt_path
                    )
                    auth = box_sdk_gen.BoxJWTAuth(config=jwt_config)

                    self.box_client = box_sdk_gen.BoxClient(
                        auth=auth
                    ).with_extra_headers(extra_headers=self.custom_header)

                    if self.box_user_id is not None:
                        user_auth = auth.with_user_subject(self.box_user_id)
                        self.box_client = box_sdk_gen.BoxClient(
                            auth=user_auth
                        ).with_extra_headers(extra_headers=self.custom_header)

                except box_sdk_gen.BoxSDKError as bse:
                    raise RuntimeError(
                        f"Error getting client from jwt token: {bse.message}"
                    )
                except Exception as ex:
                    raise ValueError(
                        "Error authenticating. Please verify your JWT config \
                            and try again."
                    ) from ex

            case "ccg":
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

                    self.box_client = box_sdk_gen.BoxClient(
                        auth=auth
                    ).with_extra_headers(extra_headers=self.custom_header)

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

            case _:
                raise ValueError(
                    f"{self.auth_type} is not a valid auth_type. Value must be \
                TOKEN, CCG, or JWT."
                )

    def get_client(self) -> box_sdk_gen.BoxClient:
        """Instantiate the Box SDK."""
        if self.box_client is None:
            self.authorize()

        return self.box_client


class BoxAPIWrapper(BaseModel):
    """Wrapper for Box API."""

    """String containing the Box Developer Token generated in the developer console"""
    box_developer_token: Optional[str] = None
    """Configured langchain_box.utilities.BoxAuth object"""
    box_auth: Optional[BoxAuth] = None
    """character_limit is an int that caps the number of characters to
       return per document."""
    character_limit: Optional[int] = -1

    box: Optional[box_sdk_gen.BoxClient]
    file_count: int = 0

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = "allow"

    @root_validator(allow_reuse=True)
    def validate_box_api_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["box"] = None

        """Validate that TOKEN auth type provides box_developer_token."""
        if not values.get("box_auth"):
            if not get_from_dict_or_env(
                values, "box_developer_token", "BOX_DEVELOPER_TOKEN"
            ):
                raise ValueError(
                    "You must configure either box_developer_token of box_auth"
                )
        else:
            box_auth = values.get("box_auth")
            values["box"] = box_auth.get_client()  #  type: ignore[union-attr]

        return values

    def get_box_client(self) -> box_sdk_gen.BoxClient:
        box_auth = BoxAuth(
            auth_type=BoxAuthType.TOKEN, box_developer_token=self.box_developer_token
        )

        self.box = box_auth.get_client()

    def _do_request(self, url: str) -> Any:
        try:
            access_token = self.box.auth.retrieve_token().access_token  #  type: ignore[union-attr]
        except box_sdk_gen.BoxSDKError as bse:
            raise RuntimeError(f"Error getting client from jwt token: {bse.message}")

        resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
        resp.raise_for_status()
        return resp.content

    def get_folder_items(self, folder_id: str) -> box_sdk_gen.Items:
        """Get all the items in a folder. Accepts folder_id as str.
        returns box_sdk_gen.Items"""
        if self.box is None:
            self.get_box_client()

        try:
            folder_contents = self.box.folders.get_folder_items(  #  type: ignore[union-attr]
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

    def get_text_representation(self, file_id: str = "") -> tuple[str, str, str]:
        try:
            from box_sdk_gen import BoxAPIError, BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")

        if self.box is None:
            self.get_box_client()

        try:
            file = self.box.files.get_file_by_id(  #  type: ignore[union-attr]
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

                if self.character_limit > 0:  #   type: ignore[operator]
                    content = raw_content[0 : self.character_limit]
                else:
                    content = raw_content

                return file_name, content, url

        return None, None, None  #   type: ignore[return-value]

    def get_document_by_file_id(self, file_id: str) -> Optional[Document]:
        """Load a file from a Box id. Accepts file_id as str.
        Returns `Document`"""

        if self.box is None:
            self.get_box_client()

        file = self.box.files.get_file_by_id(  #  type: ignore[union-attr]
            file_id, fields=["name", "type", "extension"]
        )

        if file.type == "file":
            if hasattr(DocumentFiles, file.extension.upper()):
                file_name, content, url = self.get_text_representation(file_id=file_id)

                if file_name is None or content is None or url is None:
                    return None

                metadata = {
                    "source": f"{url}",
                    "title": f"{file_name}",
                }

                return Document(page_content=content, metadata=metadata)

            return None

        return None
