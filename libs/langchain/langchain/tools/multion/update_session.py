import base64
from email.message import EmailMessage
from typing import List, Optional, Type,Any

from pydantic import BaseModel, Field
from enum import Enum
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.multion.base import MultionBaseTool

import multion
import json




class UpdateSessionSchema(BaseModel):
    """Input for UpdateSessionTool."""
    tabId:int = Field(
        ...,
        description="The tabID, received from one of the createSessions run before"
    )
    query: str = Field(
        ...,
        description="The query to run in multion agent.",
    )
    url: str = Field(
        "https://www.google.com/",
        description="The Url to run the agent at. Note: accepts only secure links having https://",
    )
    
    


class MultionUpdateSession(MultionBaseTool):
    name: str = "update_multion_session"
    description: str = (
        "Use this tool to update a existing corresponding Multion Browser Window with provided fields. Note:TabId is got from one of the previous Browser window creation."
    )
    args_schema: Type[UpdateSessionSchema] = UpdateSessionSchema
    tabId:Any = None

   
    def _run(
         self,
         tabId:str,
        query: str,
        url: Optional[str] = "https://www.google.com/",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        try:
            try:
                response = multion.update_session(tabId,{"input": query,"url": url})
                content =  {"tabId":tabId,"Response":response["message"]}
                self.tabId = tabId
                return content
            except:
                 if(self.tabID!=None):
                    response = multion.update_session(tabId,{"input": query,"url": url})
                    content =  {"tabId":self.tabId,"Response":response["message"]}
                    return content
                                     
                 else:
                    response = multion.new_session({"input": query,"url": url})
                    self.tabID= response['tabId']
                    return {"tabId":response['tabId'],"Response":response["message"]}
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
