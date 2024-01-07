from typing import List, Optional, Type, Union

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool, Tool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun
)

import subprocess
import sys


class PackageInstallInput(BaseModel):
    """Arguments for the PackageInstallTool."""

    package_names: Union[str, List[str]] = Field(
        ...,
        description="List of package name(s) to install",
        examples=[
            "pandas",
            ['pandas', 'numpy']
        ]
    )


class PackageInstallTool(BaseTool):
    """Tool that installs Python packages in runtime."""

    name: str = "install_package"
    args_schema: Type[BaseModel] = PackageInstallInput
    description: str = "Install Python packages"

    def _run(
            self,
            package_names: Union[str, List[str]],
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> bool:
        try:
            if isinstance(package_names, str):
                package_names = [package_names]
            subprocess.check_call([sys.executable, "-m", "pip", "install", *package_names])
            print(f"Packages successfully installed: {', '.join(package_names)}.")
            return True
        except Exception as e:
            print("Error: " + str(e))
            return False

    async def _arun(
            self,
            package_names: List[str],
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("install_package does not support async")

    def as_tool(self) -> Tool:
        return Tool.from_function(
            func=self._run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )
