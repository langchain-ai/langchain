from typing import TYPE_CHECKING, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

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

    sessionId: str = Field(
        ...,
        description="""The sessionID, 
        received from one of the createSessions run before""",
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
    """Tool that updates an existing Multion Browser Window with provided fields.

    Attributes:
        name: The name of the tool. Default: "update_multion_session"
        description: The description of the tool.
        args_schema: The schema for the tool's arguments. Default: UpdateSessionSchema
    """

    name: str = "update_multion_session"
    description: str = """Use this tool to update \
an existing corresponding Multion Browser Window with provided fields. \
Note: sessionId must be received from previous Browser window creation."""
    args_schema: Type[UpdateSessionSchema] = UpdateSessionSchema
    sessionId: str = ""

    def _run(
        self,
        sessionId: str,
        query: str,
        url: Optional[str] = "https://www.google.com/",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        try:
            try:
                response = multion.update_session(
                    sessionId, {"input": query, "url": url}
                )
                content = {"sessionId": sessionId, "Response": response["message"]}
                self.sessionId = sessionId
                return content
            except Exception as e:
                print(f"{e}, retrying...")
                return {"error": f"{e}", "Response": "retrying..."}
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
