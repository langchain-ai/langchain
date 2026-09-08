from typing import Any, Literal

from langchain_core.messages import ToolMessage


class RedactionRule:
    def __init__(self, pattern: str = "", replacement: str = "[REDACTED]"):
        self.pattern = pattern
        self.replacement = replacement


class DockerExecutionPolicy:
    def __init__(self, max_output_bytes: int = 10000):
        self.max_output_bytes = max_output_bytes


class CommandResult:
    def __init__(
        self,
        output: str = "",
        exit_code: int = 0,
        truncated_by_bytes: bool = False,
        truncated_by_lines: bool = False,
        total_lines: int = 1,
        total_bytes: int = 0,
    ):
        self.output = output
        self.exit_code = exit_code
        self.truncated_by_bytes = truncated_by_bytes
        self.truncated_by_lines = truncated_by_lines
        self.total_lines = total_lines
        self.total_bytes = total_bytes or len(output.encode("utf-8"))


class ShellSession:
    def __init__(
        self,
        execution_policy: DockerExecutionPolicy | None = None,
        workspace_path: Any = None,
        environment: Any = None,
        *args,
        **kwargs,
    ):
        self._execution_policy = execution_policy or DockerExecutionPolicy()
        self.workspace_path = workspace_path
        self.environment = environment

    def start(self, *args, **kwargs):
        """Starts the shell session or container."""
        return self

    def stop(self, *args, **kwargs):
        """Stops and cleans up the shell session or container."""

    def run(self, command: str, tool_call_id: str | None = None, *args, **kwargs) -> Any:
        """Executes a command and returns the formatted output string or ToolMessage."""
        output = f"Executed: {command}"
        result = CommandResult(output=output, exit_code=0)
        return self._format_output(result, tool_call_id=tool_call_id)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _format_output(self, result: Any, tool_call_id: str | None = None) -> ToolMessage | str:
        sanitized_output = getattr(result, "output", "")

        if getattr(result, "truncated_by_bytes", False):
            sanitized_output = (
                f"{sanitized_output.rstrip()}\n\n"
                f"... Output truncated at {self._execution_policy.max_output_bytes} bytes."
            )

        status_prefix = ""
        exit_code = getattr(result, "exit_code", 0)
        if exit_code not in {0, None}:
            status_prefix = f"[exit code: {exit_code}]\n"

        content = f"{status_prefix}{sanitized_output}"
        return self._format_tool_message(
            content,
            tool_call_id,
            status="success" if exit_code in {0, None} else "error",
            artifact={
                "timed_out": False,
                "exit_code": exit_code,
                "truncated_by_lines": getattr(result, "truncated_by_lines", False),
                "truncated_by_bytes": getattr(result, "truncated_by_bytes", False),
                "total_lines": getattr(result, "total_lines", 1),
                "total_bytes": getattr(result, "total_bytes", len(content.encode("utf-8"))),
                "matches": getattr(result, "matches", None),
            },
        )

    @staticmethod
    def _format_tool_message(
        content: str,
        tool_call_id: str | None,
        *,
        status: Literal["success", "error"],
        artifact: dict[str, Any] | None = None,
    ) -> ToolMessage | str:
        if tool_call_id is None:
            return content
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            status=status,
            artifact=artifact,
        )


class ShellToolMiddleware:
    def __init__(self, *args, **kwargs):
        pass
