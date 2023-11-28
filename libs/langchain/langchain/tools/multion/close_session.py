import asyncio
from typing import TYPE_CHECKING, Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
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


class CloseSessionSchema(BaseModel):
    """Input for UpdateSessionTool."""

    sessionId: str = Field(
        ...,
        description="""The sessionId, received from one of the createSessions 
        or updateSessions run before""",
    )


class MultionCloseSession(BaseTool):
    """Tool that closes an existing Multion Browser Window with provided fields.

    Attributes:
        name: The name of the tool. Default: "close_multion_session"
        description: The description of the tool.
        args_schema: The schema for the tool's arguments. Default: UpdateSessionSchema
    """

    name: str = "close_multion_session"
    description: str = """Use this tool to close \
an existing corresponding Multion Browser Window with provided fields. \
Note: SessionId must be received from previous Browser window creation."""
    args_schema: Type[CloseSessionSchema] = CloseSessionSchema
    sessionId: str = ""

    def _run(
        self,
        sessionId: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> None:
        try:
            try:
                multion.close_session(sessionId)
            except Exception as e:
                print(f"{e}, retrying...")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

    async def _arun(
        self,
        sessionId: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._run, sessionId)
