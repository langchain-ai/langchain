from typing import Any, Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool

from langchain_experimental.utilities.gigapython import (
    CodeExecutionResult,
    GigaPythonREPL,
)
import backend_bytesio


class CodeInput(BaseModel):
    code: str = Field(..., description="Код Python")


def _get_default_python_repl() -> GigaPythonREPL:
    return GigaPythonREPL(_globals=globals(), _locals=None)


def _make_message(code_result: CodeExecutionResult) -> str:
    result = code_result["result"].strip()
    if code_result["is_timeout"]:
        return (
            "Время исполнения кода было слишком долгим. "
            "Если возможно оптимизируй код, если нет, то напиши, что это сделать невозможно."  # noqa: E501
        )
    if code_result["is_exception"]:
        return (
            f'Во время исполнения кода произошла ошибка: "{result}". '
            "Исправь код и напиши исправленный код"
        )
    if len(backend_bytesio.images):
        result += "\nСгенерировано изображение\n"
    return (
        f'Результат выполнения кода: "{result}". '
        "Если ты считаешь код правильным, то отформатируй результат выполнения кода и выведи его."  # noqa: E501
    )


class GigaPythonREPLTool(BaseTool):
    """Tool for running python code in a REPL."""

    name: str = "python"
    description: str = (
        "Компилятор ipython. Возвращает результат выполнения. "
        "Если произошла ошибка напиши исправленный код."
    )

    python_repl: GigaPythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool."""
        return _make_message(self.python_repl.run(query))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        return await run_in_executor(None, self.run, query)
