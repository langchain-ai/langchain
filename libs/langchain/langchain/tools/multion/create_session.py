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
    name: str = "create_multion_session"
    description: str = """Use this tool to create a new Multion Browser Window \
        with provided fields.Always the first step to run \
            any activities that can be done using browser."""
    args_schema: Type[CreateSessionSchema] = CreateSessionSchema

    def _run(
        self,
        query: str,
        url: Optional[str] = "https://www.google.com/",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        try:
            response = multion.new_session({"input": query, "url": url})
            return {"tabId": response["tabId"], "Response": response["message"]}
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
