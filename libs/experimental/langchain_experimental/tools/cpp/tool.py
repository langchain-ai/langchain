import os
import re
import subprocess
import tempfile
from typing import Any, Optional, Type

from langchain.tools.base import BaseTool
from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, Field


def sanitize_cpp_input(code: str) -> str:
    """Sanitize input for subprocess

    This function prepares a string of C++ code for safe execution by a subprocess by:
    - Removing backticks (`) to prevent potential command execution
    - Trimming leading and trailing whitespace to clean the input

    Args:
        code (str): The code to sanitize

    Returns:
        str: The sanitized code
    """
    # Removes backticks and leading/trailing whitespace
    code = re.sub(r"`", "", code)
    code = code.strip()
    return code


class CppInputs(BaseModel):
    """
    Model for C/C++ inputs

    Attributes:
        code (str): C/C++ code snippet to run
        std (str): C++ standard to use (e.g., c++11, c++14, c++17, c++20)
        language (str): Programming language (either 'c' or 'cpp')
        cpu_limit (Optional[int]): Optional CPU time limit in seconds, default 15
        allow_dangerous_code (bool): Must be set to True to allow code execution
    """

    code: str = Field(description="C/C++ code snippet to run")
    std: str = Field(
        default="c++14",
        description="C++ standard to use (e.g., c++11, c++14, c++17, c++20)",
    )
    language: str = Field(
        default="cpp", description="Programming language (either 'c' or 'cpp')"
    )
    cpu_limit: Optional[int] = Field(
        default=15, description="Optional CPU time limit in seconds"
    )
    allow_dangerous_code: bool = Field(
        default=False, description="Must be set to True to allow code execution"
    )


class CppSubprocessTool(BaseTool):
    """
    Tool for running C/C++ code using subprocess and g++/gcc

    WARN: This tool can execute arbitrary code on the host machine
    Use with caution and only if you understand the security risks

    Attributes:
        name (str): Name of the tool
        description (str): Description of the tool
        args_schema (Type[BaseModel]): Schema for the arguments
        sanitize_input (bool): Flag to enable input sanitization
        allow_dangerous_code (bool): Flag to enable code execution
    """

    name: str = "SubprocessCpp"
    description: str = (
        "A C/C++ interpreter using subprocess and g++/gcc. "
        "Use this to execute C/C++ commands. "
        "Input should be a valid C/C++ code snippet. "
        "If you want to see the output of a value, you should print it out"
        "with `std::cout << ...` or `printf(...)`."
    )
    args_schema: Type[BaseModel] = CppInputs  # type: ignore
    sanitize_input: bool = True
    allow_dangerous_code: bool = False

    def __init__(self, allow_dangerous_code: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.allow_dangerous_code = allow_dangerous_code

    def _run(
        self,
        code: str,
        std: str = "c++14",
        language: str = "cpp",
        cpu_limit: Optional[int] = 15,
        **kwargs: Any,
    ) -> Any:
        """
        Use the tool to run C/C++ code

        WARN: This tool can execute arbitrary code on the host machine
        Use with caution and only if you understand the security risks

        Args:
            code (str): C/C++ code snippet to run
            std (str): C++ standard to use (e.g., c++11, c++14, c++17, c++20)
            language (str): Programming language (either 'c' or 'cpp')
            cpu_limit (Optional[int]): Optional CPU time limit in seconds, default 15

        Returns:
            Any: The result of the code execution.
        """
        if not self.allow_dangerous_code:
            raise PermissionError(
                "Execution of C/C++ code is disabled by default. "
                "To enable it, set allow_dangerous_code to True. "
                "Be aware that running arbitrary code can be dangerous."
            )
        try:
            if self.sanitize_input:
                code = sanitize_cpp_input(code)

            if language not in ["c", "cpp"]:
                raise ValueError(f"Invalid language specified: {language}")

            valid_standards = ["c++11", "c++14", "c++17", "c++20"]
            if std not in valid_standards:
                raise ValueError(f"Invalid C++ standard specified: {std}")

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

                with open(source_file_path, "w") as source_file:
                    source_file.write(code)

                compile_result = subprocess.run(
                    [compiler, std_flag, source_file_path, "-o", binary_file_path],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                )

                if compile_result.returncode != 0:
                    return f"Compilation failed: {compile_result.stderr}"

                def limit_resources() -> None:
                    if cpu_limit is not None:
                        import resource

                        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

                run_result = subprocess.run(
                    [binary_file_path],
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    preexec_fn=limit_resources,
                    timeout=cpu_limit,
                )

                if run_result.returncode != 0:
                    return f"Execution failed: {run_result.stderr}"

                return run_result.stdout
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {cpu_limit} seconds"
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))

    async def _arun(
        self,
        code: str,
        std: str = "c++14",
        language: str = "cpp",
        cpu_limit: Optional[int] = 15,
        **kwargs: Any,
    ) -> Any:
        """
        Use the tool to run C++ code asynchronously

        WARN: This tool can execute arbitrary code on the host machine
        Use with caution and only if you understand the security risks

        Args:
            code (str): C/C++ code snippet to run
            std (str): C++ standard to use (e.g., c++11, c++14, c++17, c++20)
            language (str): Programming language (either 'c' or 'cpp')
            cpu_limit (Optional[int]): Optional CPU time limit in seconds

        Returns:
            Any: The result of the code execution.
        """
        return await run_in_executor(
            None, self._run, code, std, language, cpu_limit, **kwargs
        )
