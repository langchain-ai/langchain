"""Util that calls Box APIs."""
from enum import Enum
import json
import requests
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, ConfigDict, validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader

"""
    AuthType - an enum to tell BoxLoader how you wish to autheticate your Box connection.

    Options are:
    TOKEN - Use a developer token generated from the Box Deevloper Token. Only recommended for development.
            provide `box_developer_token`.
    CCG - Client Credentials Grant.
          provide `box_client_id`, `box_client_secret`, `box_enterprise_id` and optionally `box_user_id`.
    JWT - Use JWT for authentication. Config should be stored on the file system accessible to your app.
          provide `box_jwt_path`. 
"""
class BoxAuthType(Enum):
    TOKEN = "token"
    CCG = "ccg"
    JWT = "jwt"

class BoxClient():

    def __init__(self,
        auth_type: str,
        box_developer_token: Optional[str] = None,
        box_client_id: Optional[str] = None,
        box_client_secret: Optional[str] = None,
        box_user_id: Optional[str] = None,
        box_enterprise_id: Optional[str] = None,
        box_jwt_path: Optional[str] = None         ,
    ):
        """Create a Box client."""
        match auth_type:
            case "token":
                try:
                    from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth, BoxSDKError
                except ImportError:
                    raise ImportError("You must run `pip install box-sdk-gen`")

                try:
                    auth = BoxDeveloperTokenAuth(token=box_developer_token)
                    self.box_client = BoxClient(auth=auth)
                except BoxSDKError as bse:
                    raise RuntimeError(f"Error getting client from developer token: {bse.message}")
                except Exception as ex:
                    raise ValueError(
                        f"Invalid Box developer token. Please verify your token and try again.\n{ex}"
                    ) from ex

            case "jwt":
                try:
                    from box_sdk_gen import BoxClient, BoxJWTAuth, JWTConfig, BoxSDKError
                except ImportError:
                    raise ImportError("You must run `pip install box-sdk-gen[jwt]`")

                try:
                    jwt_config = JWTConfig.from_config_file(config_file_path=box_jwt_path)
                    auth = BoxJWTAuth(config=jwt_config)

                    if box_user_id:
                        auth.with_user_subject(box_user_id)
                    
                    self.box_client = BoxClient(auth=auth)
                except BoxSDKError as bse:
                    raise RuntimeError(f"Error getting client from jwt token: {bse.message}")
                except Exception as ex:
                    raise ValueError(
                        "Error authenticating. Please verify your JWT config and try again."
                    ) from ex

            case "ccg":
                try:
                    from box_sdk_gen import BoxClient, BoxCCGAuth, CCGConfig, BoxSDKError
                except ImportError:
                    raise ImportError("You must run `pip install box-sdk-gen`")

                try:
                    ccg_config = CCGConfig(
                        client_id=box_client_id,
                        client_secret=box_client_secret,
                        enterprise_id=box_enterprise_id,
                    )
                    auth = BoxCCGAuth(config=ccg_config)

                    if box_user_id:
                        auth.with_user_subject(box_user_id)

                    self.box_client = BoxClient(auth=auth)
                except BoxSDKError as bse:
                    raise RuntimeError(f"Error getting client from ccg token: {bse.message}")
                except Exception as ex:
                    raise ValueError(
                        "Error authenticating. Please verify you are providing a valid client id, secret \
                            and either a valid user ID or enterprise ID."
                    ) from ex
                
            case _:
                raise ValueError(f"{self.auth_type} is not a valid auth_type. Value must be \
                TOKEN, CCG, or JWT.")
        
    def get_client(self):
        return self.box_client
    
class BoxAPIWrapper(BaseModel):
    """Wrapper for Box API."""

    model_config = ConfigDict(use_enum_values=True)
    
    auth_type: str
    box_developer_token: Optional[str] = None
    box_client_id: Optional[str] = None
    box_client_secret: Optional[str] = None
    box_user_id: Optional[str] = None
    box_enterprise_id: Optional[str] = None
    box_jwt_path: Optional[str] = None
    box_file_id: Optional[str] = None
    box_file_ids: Optional[List[str]] = None
    box_folder_id: Optional[str] = None
    box_search_query: Optional[str] = None
    box_metadata_query: Optional[str] = None
    box_metadata_template: Optional[str] = None
    box_metadata_params: Optional[str] = None
    box_ai_prompt: Optional[str] = None
    
    @root_validator()
    def validate_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:

        box = None

        """Validate auth_type is set"""
        if not values.get("auth_type"):
            raise ValueError(f"Auth type must be set.")
            
        """Validate that TOKEN auth type provides box_developer_token."""
        if (
            values.get("auth_type") == "token" and 
            not get_from_dict_or_env(values, "box_developer_token", "BOX_DEVELOPER_TOKEN")
        ):
            raise ValueError(f"{values.get('auth_type')} requires box_developer_token to be set")

        """Validate that JWT auth type provides box_jwt_path."""
        if (
            values.get("auth_type") == "jwt" and 
            not get_from_dict_or_env(values, "box_jwt_path", "BOX_JWT_PATH")
        ):
            raise ValueError(f"{values.get('auth_type')} requires box_jwt_path to be set")
        
        """Validate that CCG auth type provides box_client_id and box_client_secret and either 
        box_enterprise_id or box_user_id."""
        if values.get("auth_type") == "ccg":
            if (
                not get_from_dict_or_env(values, "box_client_id", "BOX_CLIENT_ID") or 
                not get_from_dict_or_env(values, "box_client_secret", "BOX_CLIENT_SECRET") or 
                not get_from_dict_or_env(values, "box_enterprise_id", "BOX_ENTERPRISE_ID")
            ): 
                raise ValueError(f"{values.get('auth_type')} requires box_client_id, box_client_secret, and box_enterprise_id.")

        box = BoxClient(
            auth_type = values.get("auth_type"),
            box_developer_token = values.get("box_developer_token"),
            box_client_id = values.get("box_client_id"),
            box_client_secret = values.get("box_client_secret"),
            box_user_id = values.get("box_user_id"),
            box_enterprise_id = values.get("box_enterprise_id"),
            box_jwt_path =  values.get("box_jwt_path")
        )

        values["box"] = box.get_client()
        values["TOKEN_LIMIT"] = 10000

        return values
    
    def _do_request(self, url: str):
        try:
            from box_sdk_gen import BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")
        
        try:
            access_token = self.box.auth.retrieve_token().access_token
        except BoxSDKError as bse:
            raise RuntimeError(f"Error getting client from jwt token: {bse.message}")

        resp = requests.get(
            url, headers={"Authorization": f"Bearer {access_token}"}
        )
        resp.raise_for_status()
        return resp.content    
    
    def get_folder_information(self, folder_id: str):
        try:
            from box_sdk_gen import BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")
        
        try:
            folder_contents = self.box.folders.get_folder_items(folder_id, fields=["name","id"])
        except BoxSDKError as bse:
            raise RuntimeError(f"Error getting client from jwt token: {bse.message}")

        return folder_contents.entries
        
    def get_text_representation(self, query: str = None, file_id: str = None):

        try:
            from box_sdk_gen import BoxSDKError, BoxAPIError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")
        
        if file_id is None:
            file_id=self.box_file_id

        try:
            file = self.box.files.get_file_by_id(file_id, x_rep_hints="[extracted_text]", fields=["name","representations"])
        except BoxSDKError as sdk_ex:
            print(f"BoxSDKError: Error getting text rep: {sdk_ex.message}")
            return None, None, None
        except BoxAPIError as api_ex:
            print(f"BoxAPIError: Error getting text rep: {api_ex.message}")
            return None, None, None
        except Exception as ex:
            print(f"Exception: Error getting text rep: {ex.message}")
            return None, None, None

        file_repr = file.representations.entries

        if len(file_repr) <=0:
            print(f"GTR: no entries for file {file_id}\n\n")
            return None, None, None

        for entry in file_repr:

           if entry.representation == "extracted_text":

                # If the file representation doesn't exist, calling info.url will generate text if possible
                if entry.status.state == "none":
                    self._do_request(entry.info.url)

                url = entry.content.url_template.replace("{+asset_path}", "")
                file_name = file.name.replace(".", "_").replace(" ", "_")

                raw_content = self._do_request(url)

                content = raw_content[0:self.TOKEN_LIMIT]

                return file_name, content, url

    def get_document_by_file_id(self, file_id: str) -> Optional[Document]:
        """Load a file from a Box id."""
        
        file_name, content, url = self.get_text_representation(file_id=file_id)
        
        if file_name is None or content is None or url is None:
            print("No text representation available for file {file_id}. Skipping...")
            return None
       
        metadata = {
            "source": f"{url}",
            "title": f"{file_name}",
        }

        return Document(page_content=content, metadata=metadata)
                
    def get_documents_by_file_ids(self, box_file_ids: List[str] = None) -> List[Document]:
        """Load documents from a list of Box file paths."""

        if box_file_ids is None:
            box_file_ids = self.box_file_ids
            
        
        files = []

        for file_id in box_file_ids:
            file = self.get_document_by_file_id(file_id)
            
            if file is not None:
                files.append(file)

        return files
    
    def get_documents_by_folder_id(self, box_folder_id: str = None) -> List[Document]:

        if box_folder_id is None:
            box_folder_id = self.box_folder_id
            
        """Load documents from a Box folder."""
        folder_content = self.get_folder_information(box_folder_id)
        
        files = []
        
        for file in folder_content:
            file = self.get_document_by_file_id(file.id)
            if file is not None:
                files.append(file)

        return files
    
    def get_search_results(self, query:str = None):
        
        try:
            from box_sdk_gen import BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")
        
        if query is None:
            query = self.box_search_query
        
        files = []
        try:
            results = self.box.search.search_for_content(query=query, fields=["id"])

            for file in results.entries:
                if file is not None:
                    files.append(file.id)

            return files
        except BoxSDKError as bse:
            raise RuntimeError(f"Error getting search results: {bse.message}")

        
    
    def get_documents_by_search(self, query: str = None) -> List[Document]:
        
        if query is None:
            query = self.box_search_query
        
        print(f"GDBS: query {query} token {self.box_developer_token}")
        files = self.get_search_results(query)

        if files is None or len(files) <= 0:
            return("no files found")
        
        print(f"GDBS: files {files}")
        
        return self.get_documents_by_file_ids(files)
    
    def get_metadata_query_results(self, query: str = None, template: str = None, param_string: str = None, eid: str = None):
        try:
            from box_sdk_gen import BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")
        
        if query is None:
            query = self.box_metadata_query

        if template is None:
            template = self.box_metadata_template

        if param_string is None:
            param_string = self.box_metadata_params

        if eid is None:
            eid = self.box_enterprise_id
        
        files = []
        params = json.loads(param_string)

        try:
            results = self.box.search.search_by_metadata_query(f"enterprise_{eid}.{template}", ancestor_folder_id="0", query=query, query_params=params)
        except BoxSDKError as bse:
            raise RuntimeError(f"Error getting metadata_query results: {bse.message}")
        
        for file in results.entries:
            if file is not None:
                files.append(file.id)

        return files
    
    def get_documents_by_metadata_query(self, query: str = None, template: str = None, param_string: str = None, eid: str = None) -> List[Document]:
        
        try:
            from box_sdk_gen import BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")
        
        if query is None:
            query = self.box_metadata_query

        if template is None:
            template = self.box_metadata_template

        if param_string is None:
            param_string = self.box_metadata_params

        if eid is None:
            eid = self.box_enterprise_id
        
        files = self.get_metadata_query_results(query, template, param_string, eid)

        if len(files) <= 0:
            return("no files found")
        
        return self.get_documents_by_file_ids(files)
    
    def get_documents_by_box_ai_ask(self, query: str = None, file_ids: List[str] = None, return_response: bool = False) -> List[Document]:

        if query is None:
            query = self.box_ai_prompt
        if file_ids is None:
            file_ids = self.box_file_ids

        try:
            from box_sdk_gen import CreateAiAskMode, CreateAiAskItems, CreateAiAskItemsTypeField, BoxSDKError
        except ImportError:
            raise ImportError("You must run `pip install box-sdk-gen`")
        
        ai_mode = CreateAiAskMode.SINGLE_ITEM_QA.value

        if len(file_ids) > 1:
            ai_mode = CreateAiAskMode.MULTIPLE_ITEM_QA.value
        elif len(file_ids) <= 0:
            raise ValueError("BOX_AI_ASK requires at least one file ID")
        
        items = []

        for file_id in file_ids:
            item = CreateAiAskItems(
                id=file_id,
                type=CreateAiAskItemsTypeField.FILE.value
            )
            items.append(item)

        try: 
            response = self.box.ai.create_ai_ask(
                ai_mode,
                query,
                items
            )
        except BoxSDKError as bse:
            raise RuntimeError(f"Error getting Box AI Response: {bse.message}")

        content=response.answer

        if return_response:
            return content
        
        metadata = {
            "source": f"Box AI",
            "title": f"Box AI {query}"
        }

        return [Document(page_content=content, metadata=metadata)]
    
    def get_documents_by_box_ai_extract(self, query: str = None, file_ids: List[str] = None) -> List[Document]:

        if query is None:
            query = self.box_ai_prompt
        if file_ids is None:
            file_ids = self.box_file_ids
            
        return NotImplemented
    
    def get_documents_by_citations(self, query: str  = None, file_ids: List[str] = None):

        if query is None:
            query = self.box_ai_prompt
        if file_ids is None:
            file_ids = self.box_file_ids
            
        return NotImplemented

    def run(self, mode: str, query: str) -> str:
        match mode:
            case "search":
                return self.search(query)
            case "metadata_query":
                return self.metadata_query(query)
            case "box_ai_ask":
                return self.box_ai_ask_response(query)
            case "box_ai_extract":
                return self.box_ai_exxtract_response(query)
            case "citations":
                return self.citations(query)
            case _:
                raise ValueError(f"Got unexpected mode {mode}")
