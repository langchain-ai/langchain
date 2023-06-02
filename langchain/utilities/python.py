from typing import Dict, Optional

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from wasm_exec import WasmExecutor


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


def _get_wasm_executor() -> WasmExecutor:
    return WasmExecutor()


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")
    wasm: WasmExecutor = Field(default_factory=_get_wasm_executor)

    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        try:
            print(self.globals)
            result = self.wasm.exec(command, globals=self.globals, locals=self.locals)
            output = result.text
        except Exception as e:
            output = repr(e)
        return output
