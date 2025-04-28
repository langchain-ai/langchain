from typing import TYPE_CHECKING, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    import multion
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        import multion
    except ImportError:
        pass


class CreateSessionSchema(BaseModel):
    """Input for CreateSessionTool."""

    query: str = Field(
        ...,
        description="The query to run in multion agent.",
    )
    url: str = Field(
        "https://www.google.com/",
        description="""The Url to run the agent at. Note: accepts only secure \
            links having https://""",
    )


class MultionCreateSession(BaseTool):
    """Tool that creates a new Multion Browser Window with provided fields.

    Attributes:
        name: The name of the tool. Default: "create_multion_session"
        description: The description of the tool.
        args_schema: The schema for the tool's arguments.
    """

    name: str = "create_multion_session"
    description: str = """
        Create a new web browsing session based on a user's command or request. \
        The command should include the full info required for the session. \
        Also include an url (defaults to google.com if no better option) \
        to start the session. \
        Use this tool to create a new Browser Window with provided fields. \
        Always the first step to run any activities that can be done using browser.
        """
    args_schema: Type[CreateSessionSchema] = CreateSessionSchema

    def _run(
        self,
        query: str,
        url: Optional[str] = "https://www.google.com/",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        try:
            response = multion.new_session({"input": query, "url": url})
            return {
                "sessionId": response["session_id"],
                "Response": response["message"],
            }
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
