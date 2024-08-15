from typing import Any, Dict, Iterator, List, Optional

from box_sdk_gen import FileBaseTypeField  # type: ignore
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, ConfigDict, root_validator

from langchain_box.utilities import BoxAPIWrapper, BoxAuth


class BoxLoader(BaseLoader, BaseModel):
    """
    BoxLoader

    This class will help you load files from your Box instance. You must have a
    Box account. If you need one, you can sign up for a free developer account.
    You will also need a Box application created in the developer portal, where
    you can select your authorization type.

    If you wish to use either of the Box AI options, you must be on an Enterprise
    Plus plan or above. The free developer account does not have access to Box AI.

    In addition, using the Box AI API requires a few prerequisite steps:
    * Your administrator must enable the Box AI API
    * You must enable the `Manage AI` scope in your app in the developer console.
    * Your administratormust install and enable your application.

    Example Implementation

    ```python
    from langchain_box.document_loaders import BoxLoader
    from langchain_box.utilities import BoxAuth, BoxAuthType

    auth = BoxAuth(
        auth_type=BoxAuthType.TOKEN,
        box_developer_token=box_developer_token
    )

    loader = BoxLoader(
        box_auth=auth,
        box_file_ids=["12345", "67890"],
        character_limit=10000,  # Optional. Defaults to no limit
        get_text_rep=True,  # Get text rep first when available, default True
        get_images=False  # Download images, defaults to False
    )

    docs = loader.lazy_load()
    ```

    Initialization variables
    variable | description | type | required
    ---+---+---
    box_developer_token | token to use for auth. | string | no
    box_auth | client id for you app. Used for CCG | string | no
    box_file_ids | Array of Box file Ids to retrieve | array of strings | no
    box_folder_id | Box folder id to retrieve | string | no
    recursive | whether to return subfolders, default False | bool | no
    get_text_rep | whether to attempt to get text, default True | bool | no
    get_images | whether to download images, default False | bool | no

    Getting and parsing images relies on
    `langchain_community.document_loaders.image import UnstructuredImageLoader`

    All of the dependencies are installed when you install the langchain_box
    package, but you must have tesseract installed locally and the bin
    directory must be in the PATH variable in your OS environment.
    """

    model_config = ConfigDict(use_enum_values=True)

    """String containing the Box Developer Token generated in the developer console"""
    box_developer_token: Optional[str] = None
    """Configured langchain_box.utilities.BoxAuth object"""
    box_auth: Optional[BoxAuth] = None
    """List[str] containing Box file ids"""
    box_file_ids: Optional[List[str]] = None
    """String containing box folder id to load files from"""
    box_folder_id: Optional[str] = None
    """If getting files by folder id, recursive is a bool to determine if you wish 
       to traverse subfolders to return child documents. Default is False"""
    recursive: Optional[bool] = False
    """character_limit is an int that caps the number of characters to
       return per document."""
    character_limit: Optional[int] = -1
    """Bool that instructs langchain_box to attempt to get text representations
       when available. Defaults to True"""
    get_text_rep: Optional[bool] = True
    """Bool that instructs langchain_box to download images. Default is False,
       and images will be skipped"""
    get_images: Optional[bool] = False

    box: Optional[BoxAPIWrapper]

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @root_validator(allow_reuse=True)
    def validate_box_loader_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        box = None

        """Validate that has either box_file_ids or box_folder_id."""
        if not values.get("box_file_ids") and not values.get("box_folder_id"):
            raise ValueError("You must provide box_file_ids or box_folder_id.")

        """Validate that we don't have both box_file_ids and box_folder_id."""
        if values.get("box_file_ids") and values.get("box_folder_id"):
            raise ValueError(
                "You must provide either box_file_ids or box_folder_id, not both."
            )

        """Validate that we have either a box_developer_token or box_auth."""
        if not values.get("box_auth") and not values.get("box_developer_token"):
            raise ValueError(
                "you must provide box_developer_token or a box_auth "
                "generated with langchain_box.utilities.BoxAuth"
            )

        box = BoxAPIWrapper(  # type: ignore[call-arg]
            box_developer_token=values.get("box_developer_token"),
            box_auth=values.get("box_auth"),
            get_text_rep=values.get("get_text_rep"),
            get_images=values.get("get_images"),
            character_limit=values.get("character_limit"),
        )

        values["box"] = box

        return values

    def _get_files_from_folder(self, folder_id):  # type: ignore[no-untyped-def]
        folder_content = self.box.get_folder_items(folder_id)

        for file in folder_content:
            if file.type == FileBaseTypeField.FILE:
                doc = self.box.get_document_by_file_id(file.id)

                if doc is not None:
                    yield doc

            elif file.type == "folder" and self.recursive:
                try:
                    yield from self._get_files_from_folder(file.id)
                except TypeError:
                    pass

    def lazy_load(self) -> Iterator[Document]:
        """Load documents. Accepts no arguments. Returns `Iterator[Document]`"""
        if self.box_file_ids:
            for file_id in self.box_file_ids:
                file = self.box.get_document_by_file_id(file_id)  # type: ignore[union-attr]

                if file is not None:
                    yield file
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
