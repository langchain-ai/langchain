""" Tool for interacting with the Memgraph graph database. """ 

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_community.utilities.memgraph import Memgraph




class BaseMemgraphDatabaseTool(BaseModel):
    """Base tool for interacting with a Memgraph database."""

    db: Memgraph = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )



class GetSchemaMemgraphDatabaseTool(BaseMemgraphDatabaseTool, BaseTool):  # type: ignore[override, override]
    """Tool for getting the schema of a Memgraph database."""

    name: str = "memgraph_db_get_schema"
    description: str = """
    Get the schema of the Memgraph database.
    """
    args_schema: Type[BaseModel] = BaseModel

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema of the Memgraph database."""
        return self.db.get_schema()