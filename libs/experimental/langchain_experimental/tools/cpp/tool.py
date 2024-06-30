import subprocess
import tempfile
import os
import re
from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor

def sanitize_cpp_input(code: str) -> str:
    """Sanitize input for subprocess

    Remove unnecessary whitespace and backticks

    Args:
        code: The code to sanitize

    Returns:
        str: The sanitized code
    """
    # Removes backticks and leading/trailing whitespace
    code = re.sub(r"`", "", code)
    code = code.strip()
    return code

class CppInputs(BaseModel):
    """C/C++ inputs"""
    code: str = Field(description="C/C++ code snippet to run")
    std: str = Field(default="c++14", description="C++ standard to use (e.g., c++11, c++14, c++17, c++20)")
    language: str = Field(default="cpp", description="Programming language (either 'c' or 'cpp')")
    cpu_limit: Optional[int] = Field(default=None, description="Optional CPU time limit in seconds")

class CppSubprocessTool(BaseTool):
    """Tool for running C/C++ code using subprocess and g++/gcc"""
    name: str = "SubprocessCpp"
    description: str = (
        "A C/C++ interpreter using subprocess and g++/gcc. Use this to execute C/C++ commands."
        "Input should be a valid C/C++ code snippet."
        "If you want to see the output of a value, you should print it out"
        "with `std::cout << ...` or `printf(...)`."
    )
    args_schema: Type[BaseModel] = CppInputs
    sanitize_input: bool = True

    def _run(
        self,
        code: str,
        std: str = "c++14",
        language: str = "cpp",
        cpu_limit: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool to run C/C++ code"""
        try:
            if self.sanitize_input:
                code = sanitize_cpp_input(code)

            if language not in ["c", "cpp"]:
                raise ValueError(f"Invalid language specified: {language}")
                
            with tempfile.TemporaryDirectory() as temp_dir:
                if language == "c":
                    source_file_path = os.path.join(temp_dir, "temp_code.c")
                    binary_file_path = os.path.join(temp_dir, "temp_binary")
                    compiler = "gcc"
                    std_flag = f"-std={std}"
                else:
                    source_file_path = os.path.join(temp_dir, "temp_code.cpp")
                    binary_file_path = os.path.join(temp_dir, "temp_binary")
                    compiler = "g++"
                    std_flag = f"-std={std}"

                with open(source_file_path, 'w') as source_file:
                    source_file.write(code)

                compile_result = subprocess.run(
                    [compiler, std_flag, source_file_path, "-o", binary_file_path],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir
                )

                if compile_result.returncode != 0:
                    return f"Compilation failed: {compile_result.stderr}"

                def limit_resources():
                    if cpu_limit is not None:
                        import resource
                        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

                run_result = subprocess.run(
                    [binary_file_path],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    preexec_fn=limit_resources if cpu_limit is not None else None
                )

                if run_result.returncode != 0:
                    return f"Execution failed: {run_result.stderr}"

                return run_result.stdout
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))

    async def _arun(
        self,
        code: str,
        std: str = "c++14",
        language: str = "cpp",
        cpu_limit: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool to run C++ code asynchronously"""
        return await run_in_executor(None, self._run, code, std, language, cpu_limit)
