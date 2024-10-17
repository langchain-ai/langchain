from typing import Iterator, List, Optional

from box_sdk_gen import FileBaseTypeField  # type: ignore
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import from_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_box.utilities import BoxAuth, _BoxAPIWrapper


class BoxLoader(BaseLoader, BaseModel):
    """BoxLoader.

    This class will help you load files from your Box instance. You must have a
    Box account. If you need one, you can sign up for a free developer account.
    You will also need a Box application created in the developer portal, where
    you can select your authorization type.

    If you wish to use either of the Box AI options, you must be on an Enterprise
    Plus plan or above. The free developer account does not have access to Box AI.

    In addition, using the Box AI API requires a few prerequisite steps:

    * Your administrator must enable the Box AI API
    * You must enable the ``Manage AI`` scope in your app in the developer console.
    * Your administrator must install and enable your application.

    **Setup**:
        Install ``langchain-box`` and set environment variable ``BOX_DEVELOPER_TOKEN``.

        .. code-block:: bash

            pip install -U langchain-box
            export BOX_DEVELOPER_TOKEN="your-api-key"


    This loader returns ``Document`` objects built from text representations of files
    in Box. It will skip any document without a text representation available. You can
    provide either a ``List[str]`` containing Box file IDS, or you can provide a
    ``str`` contining a Box folder ID. If providing a folder ID, you can also enable
    recursive mode to get the full tree under that folder.

    .. note::
        A Box instance can contain Petabytes of files, and folders can contain millions
        of files. Be intentional when choosing what folders you choose to index. And we
        recommend never getting all files from folder 0 recursively. Folder ID 0 is your
        root folder.

    **Instantiate**:

        .. list-table:: Initialization variables
            :widths: 25 50 15 10
            :header-rows: 1

            * - Variable
              - Description
              - Type
              - Default
            * - box_developer_token
              - Token to use for auth.
              - ``str``
              - ``None``
            * - box_auth
              - client id for you app. Used for CCG
              - ``langchain_box.utilities.BoxAuth``
              - ``None``
            * - box_file_ids
              - client id for you app. Used for CCG
              - ``List[str]``
              - ``None``
            * - box_folder_id
              - client id for you app. Used for CCG
              - ``str``
              - ``None``
            * - recursive
              - client id for you app. Used for CCG
              - ``Bool``
              - ``False``
            * - character_limit
              - client id for you app. Used for CCG
              - ``int``
              - ``-1``


    **Get files** — this method requires you pass the ``box_file_ids`` parameter.
    This is a ``List[str]`` containing the file IDs you wish to index.

        .. code-block:: python

            from langchain_box.document_loaders import BoxLoader

            box_file_ids = ["1514555423624", "1514553902288"]

            loader = BoxLoader(
                box_file_ids=box_file_ids,
                character_limit=10000  # Optional. Defaults to no limit
            )

    **Get files in a folder** — this method requires you pass the ``box_folder_id``
    parameter. This is a ``str`` containing the folder ID you wish to index.

        .. code-block:: python

            from langchain_box.document_loaders import BoxLoader

            box_folder_id = "260932470532"

            loader = BoxLoader(
                box_folder_id=box_folder_id,
                recursive=False  # Optional. return entire tree, defaults to False
            )

    **Load**:
        .. code-block:: python

            docs = loader.load()
            docs[0]

        .. code-block:: python

            Document(metadata={'source': 'https://dl.boxcloud.com/api/2.0/
            internal_files/1514555423624/versions/1663171610024/representations
            /extracted_text/content/', 'title': 'Invoice-A5555_txt'},
            page_content='Vendor: AstroTech Solutions\\nInvoice Number: A5555\\n\\nLine
            Items:\\n    - Gravitational Wave Detector Kit: $800\\n    - Exoplanet
            Terrarium: $120\\nTotal: $920')

    **Lazy load**:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Document(metadata={'source': 'https://dl.boxcloud.com/api/2.0/
            internal_files/1514555423624/versions/1663171610024/representations
            /extracted_text/content/', 'title': 'Invoice-A5555_txt'},
            page_content='Vendor: AstroTech Solutions\\nInvoice Number: A5555\\n\\nLine
            Items:\\n    - Gravitational Wave Detector Kit: $800\\n    - Exoplanet
            Terrarium: $120\\nTotal: $920')

    """

    box_developer_token: Optional[str] = Field(
        default_factory=from_env("BOX_DEVELOPER_TOKEN", default=None)
    )
    """String containing the Box Developer Token generated in the developer console"""

    box_auth: Optional[BoxAuth] = None
    """Configured 
       `BoxAuth <https://python.langchain.com/v0.2/api_reference/box/utilities/langchain_box.utilities.box.BoxAuth.html>`_ 
       object"""

    box_file_ids: Optional[List[str]] = None
    """List[str] containing Box file ids"""

    box_folder_id: Optional[str] = None
    """String containing box folder id to load files from"""

    recursive: Optional[bool] = False
    """If getting files by folder id, recursive is a bool to determine if you wish 
       to traverse subfolders to return child documents. Default is False"""

    character_limit: Optional[int] = -1
    """character_limit is an int that caps the number of characters to
       return per document."""

    _box: Optional[_BoxAPIWrapper] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=True,
    )

    @model_validator(mode="after")
    def validate_box_loader_inputs(self) -> Self:
        _box = None

        """Validate that has either box_file_ids or box_folder_id."""
        if not self.box_file_ids and not self.box_folder_id:
            raise ValueError("You must provide box_file_ids or box_folder_id.")

        """Validate that we don't have both box_file_ids and box_folder_id."""
        if self.box_file_ids and self.box_folder_id:
            raise ValueError(
                "You must provide either box_file_ids or box_folder_id, not both."
            )

        """Validate that we have either a box_developer_token or box_auth."""
        if not self.box_auth:
            if not self.box_developer_token:
                raise ValueError(
                    "you must provide box_developer_token or a box_auth "
                    "generated with langchain_box.utilities.BoxAuth"
                )
            else:
                _box = _BoxAPIWrapper(  # type: ignore[call-arg]
                    box_developer_token=self.box_developer_token,
                    character_limit=self.character_limit,
                )
        else:
            _box = _BoxAPIWrapper(  # type: ignore[call-arg]
                box_auth=self.box_auth,
                character_limit=self.character_limit,
            )

        self._box = _box

        return self

    def _get_files_from_folder(self, folder_id):  # type: ignore[no-untyped-def]
        folder_content = self.box.get_folder_items(folder_id)

        for file in folder_content:
            try:
                if file.type == FileBaseTypeField.FILE:
                    doc = self._box.get_document_by_file_id(file.id)

                    if doc is not None:
                        yield doc

                elif file.type == "folder" and self.recursive:
                    try:
                        yield from self._get_files_from_folder(file.id)
                    except TypeError:
                        pass
            except TypeError:
                pass

    def lazy_load(self) -> Iterator[Document]:
        """Load documents. Accepts no arguments. Returns `Iterator[Document]`"""
        if self.box_file_ids:
            for file_id in self.box_file_ids:
                try:
                    file = self._box.get_document_by_file_id(file_id)  # type: ignore[union-attr]

                    if file is not None:
                        yield file
                except TypeError:
                    pass
        elif self.box_folder_id:
            try:
                yield from self._get_files_from_folder(self.box_folder_id)
            except TypeError:
                pass
            except Exception as e:
                print(f"Exception {e}")  # noqa: T201
        else:
            raise ValueError(
                "You must provide either `box_file_ids` or `box_folder_id`"
            )
