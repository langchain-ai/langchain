from typing import TYPE_CHECKING, Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    import multion
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        import multion
    except ImportError:
        pass


class UpdateSessionSchema(BaseModel):
    """Input for UpdateSessionTool."""

    tabId: str = Field(
        ..., description="The tabID, received from one of the createSessions run before"
    )
    query: str = Field(
        ...,
        description="The query to run in multion agent.",
    )
    url: str = Field(
        "https://www.google.com/",
        description="""The Url to run the agent at. \
        Note: accepts only secure links having https://""",
    )


class MultionUpdateSession(BaseTool):
    name: str = "update_multion_session"
    description: str = """Use this tool to update \
        a existing corresponding \
    Multion Browser Window with provided fields. \
        Note:TabId is got from one of the previous Browser window creation."""
    args_schema: Type[UpdateSessionSchema] = UpdateSessionSchema

    def _run(
        self,
        tabId: str,
        query: str,
        url: Optional[str] = "https://www.google.com/",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        try:
            try:
                response = multion.update_session(tabId, {"input": query, "url": url})
                content = {"tabId": tabId, "Response": response["message"]}
                self.tabId = tabId
                return content
            except Exception as e:
                print(f"{e}, creating a new session")
                response = multion.new_session({"input": query, "url": url})
                self.tabID = response["tabId"]
                return {"tabId": response["tabId"], "Response": response["message"]}
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
