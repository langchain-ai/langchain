from typing import Any, Dict, List, Optional
from enum import Enum

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.box import BoxAPIWrapper, BoxAuthType
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, ConfigDict, validator
from langchain_core.utils import get_from_dict_or_env



"""
    Mode - an enum to tell BoxLoader how you wish to retrieve your files.

    Options are:
    FILES - provide `[box_file_ids]`. Will return contents of each file.
    FOLDER - provide `box_folder_id`. Will return all files from that folder.
    TREE - provide `box_folder_id`. Will return everything in that folder recursively. **Use with caution**!
    SEARCH - provide `box_search_query`. Will return all files meeting criteria.
    METADATA_QUERY - provide `box_metadata_query` and `box_metadata_template`. Will return all files with matching Metadata entries.
    BOX_AI_ASK - provide `box_ai_prompt` and `[box_file_ids]`. Will return the response to your prompt.
    BOX_AI_EXTRACT - provide `box_ai_prompt` and `[box_file_ids]`. Will return the response to your prompt.
    BOX_AI_CITATIONS - provide `box_ai_prompt` and `[box_file_ids]`. Will return the citations for to your prompt.
"""
class Mode(Enum):
    FILES = "files"
    FOLDER = "folder"
    TREE = "tree"
    SEARCH = "search"
    METADATA_QUERY = "metadata_query"
    BOX_AI_ASK = "box_ai_ask"
    BOX_AI_EXTRACT = "box_ai_extract"
    CITATIONS = "citations"

class BoxLoader(BaseLoader, BaseModel):
    """
        BoxLoader
        
        This class will help you load files from your Box instance. You must have a Box account.
        If you need one, you can sign up for a free developer account. You will also need a Box 
        application created in the developer portal, where you can select your authorization type.
        If you wish to use either of the Box AI options, you must be on an Enterprise Plus plan or
        above. The free developer account does not have access to Box AI. 
        
        In addition, using the Box AI API requires a few prerequisite steps:
        * Your administrator must enable the Box AI API
        * You must enable the `Manage AI` scope in your application in the developer console. 
        * Your administratormust install and enable your application.

        Example Implementation
        ```
        ```

        Initialization variables
        variable | description | type | required
        ---+---+---
        mode | how to retrieve documents | enum | yes
        auth_type | authentication type to use | enum | yes
        box_developer_token | token to use for auth. Should only use for development | string | no
        box_client_id | client id for you app. Used for CCG | string | no
        box_client_secret | client secret for you app. Used for CCG | string | no
        box_user_id | User ID or Enterprise ID to make calls for. Used for CCG or JWT | string | no
        box_enterprise_id | Enterprise ID to make calls for. Used for CCG. | string | no
        box_jwt_path | Local file system path the the jwt config JSON | string | no
        box_file_ids | Array of Box file Ids to retrieve | array of strings | no
        box_folder_id | Id of folder to process | string | no
        box_search_query | query to search for files to retrieve | string | no
        box_metadata_query | metadata query to search for files to retrieve | string | no
        box_metadata_template | metadata template to search for files to retrieve | string | no
        box_metadata_params | params to complete the metadata query to search for files to retrieve | string | no
        box_ai_prompt | prompt to query Box AI to retrieve a response or citations | string | no
    """
    model_config = ConfigDict(use_enum_values=True)
    
    mode: Mode
    auth_type: BoxAuthType
    box_developer_token: Optional[str] = None
    box_client_id: Optional[str] = None
    box_client_secret: Optional[str] = None
    box_user_id: Optional[str] = None
    box_enterprise_id: Optional[str] = None
    box_jwt_path: Optional[str] = None
    box_file_ids: Optional[List[str]] = None
    box_folder_id: Optional[str] = None
    box_search_query: Optional[str] = None
    box_metadata_query: Optional[str] = None
    box_metadata_template: Optional[str] = None
    box_metadata_params: Optional[str] = None
    box_ai_prompt: Optional[str] = None

    @validator('mode')
    def validate_mode(cls, value):
        if value is None and hasattr(Mode,value):
            raise ValueError("You must provide a valid mode")
        
        return value.value
    
    @validator('auth_type')
    def validate_auth_type(cls, value):
        if value is None and hasattr(BoxAuthType,value):
            raise ValueError("You must provide a valid auth_type")
        
        return value.value
    
    @root_validator()
    def validate_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:

        box = None
        
        """Validate that FILES mode provides box_file_ids."""
        if values.get("mode") == "files" and not values.get("box_file_ids"):
            raise ValueError(f"{values.get('mode')} requires box_file_ids to be set")
        
        """Validate that FOLDER and TREE mode provides box_folder_id."""
        if values.get("mode") == "folder" or values.get("mode") == "tree":
            if not values.get("box_folder_id"):
                raise ValueError(f"{values.get('mode')} requires box_folder_id to be set")
            
        """Validate that SEARCH mode provides search_query."""    
        if values.get("mode") == "search" and not values.get("box_search_query"):
            raise ValueError(f"{values.get('mode')} requires search_query to be set")
        
        """Validate that METADATA_QUERY mode provides metadata_query."""
        if values.get("mode") == "metadata_query": 
            if (
                not values.get("box_metadata_query") and 
                not values.get("box_metadata_template") and 
                not values.get("box_metadata_params") and 
                not values.get("box_enterprise_id")
            ):
                raise ValueError(f"{values.get('mode')} requires box_metadata_query, box_metadata_template, box_metadata_params, and boc_enterprise_id to be set")

        """Validate that BOX_AI and CITATIONS mode provides box_ai_prompt."""
        if (
            values.get("mode") == "box_ai_ask" or 
            values.get("mode") == "box_ai_extract" or 
            values.get("mode") == "citations"
         ):
            if not values.get("box_ai_prompt"):
                raise ValueError(f"{values.get('mode')} requires box_ai_prompt to be set")
            
        """Validate that TOKEN auth type provides box_developer_token."""
        if (
            values.get("auth_type") == "token" and 
            not values.get("box_developer_token")
        ):
            raise ValueError(f"{values.get('auth_type')} requires box_developer_token to be set")
        
        """Validate that JWT auth type provides box_jwt_path."""
        if (
            values.get("auth_type") == "jwt" and 
            not values.get("box_jwt_path")
        ):
            raise ValueError(f"{values.get('auth_type')} requires box_jwt_path to be set")
        
        """Validate that CCG auth type provides box_client_id and box_client_secret and either 
        box_enterprise_id or box_user_id."""
        if values.get("auth_type") == "ccg":
            if(
                not values.get("box_client_id") or 
                not values.get("box_client_secret") or 
                not values.get("box_enterprise_id")
            ): 
                raise ValueError(f"{values.get('auth_type')} requires box_client_id, box_client_secret, and box_enterprise_id.")

        box = BoxAPIWrapper(
            auth_type=values.get("auth_type"),
            box_developer_token=values.get("box_developer_token"),
            box_client_id=values.get("box_client_id"),
            box_client_secret=values.get("box_client_secret"),
            box_enterprise_id=values.get("box_enterprise_id"),
            box_jwt_path=values.get("box_jwt_path"),
            box_user_id=values.get("box_user_id"),
            box_file_ids=values.get("box_file_ids"),
            box_folder_id=values.get("box_folder_id"),
            box_search_query=values.get("box_search_query"),
            box_metadata_query=values.get("box_metadata_query"),
            box_metadata_template=values.get("box_metadata_template"),
            box_metadata_params=values.get("box_metadata_params"),
            box_ai_prompt=values.get("box_ai_prompt")
        )

        values["box"] = box

        return values
   
    def load(self) -> List[Document]:
        """Load documents."""
        match self.mode:
            case "files":
                return self.box.get_documents_by_file_ids()
            case "folder":
                return self.box.get_documents_by_folder_id()
            case "search":
                return self.box.get_documents_by_search()
            case "metadata_query":
                return self.box.get_documents_by_metadata_query()
            case "box_ai_ask":
                return self.box.get_documents_by_box_ai_ask()
            case "box_ai_extract":
                return self.box.get_documents_by_box_ai_extract()
            case "citations":
                return self.box.get_documents_by_citations()
            case _:
                raise ValueError(f"{self.mode} is not a valid mode. Value must be \
                    FILES, FOLDER, TREE, SEARCH, METADATA_QUERY, BOX_AI, or CITATIONS.")