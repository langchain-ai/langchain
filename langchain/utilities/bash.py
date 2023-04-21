"""Wrapper around subprocess to run commands."""
import subprocess
from typing import List, Union


# Function to create a reusable Bash process
def create_bash_process():
    global pexpect
    import pexpect
    process = pexpect.spawn('bash', encoding='utf-8')
    # Set the prompt to something we can find
    process.sendline('export PS1=UNIQUE_PROMPT_ASDFFDSA')
    process.expect('UNIQUE_PROMPT_ASDFFDSA')
    return process


class BashProcess:
    """Executes bash commands and returns the output."""

    def __init__(self, strip_newlines: bool = False, return_err_output: bool = False, persistent=False):
        """Initialize with stripping newlines."""
        self.strip_newlines = strip_newlines
        self.return_err_output = return_err_output
        if persistent:
            self.process = create_bash_process()

    def run(self, commands: Union[str, List[str]]) -> str:
        """Run commands and return final output."""
        if isinstance(commands, str):
            commands = [commands]
        commands = ";".join(commands)
        if hasattr(self, 'process'):
            return self._run_persistent(commands)
        else:
            return self._run(commands)

    def _run(self, command: str) -> str:
        """Run commands and return final output."""
        try:
            output = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout.decode()
        except subprocess.CalledProcessError as error:
            if self.return_err_output:
                return error.stdout.decode()
            return str(error)
        if self.strip_newlines:
            output = output.strip()
        return output

    def _run_persistent(self, command: str) -> str:
        """Run commands and return final output."""
        self.process.sendline(command)

        # Clear the output with an empty string
        self.process.expect('UNIQUE_PROMPT_ASDFFDSA', timeout=10)
        self.process.sendline('')

        try:
            self.process.expect(['UNIQUE_PROMPT_ASDFFDSA', pexpect.EOF], timeout=10)
        except pexpect.TIMEOUT:
            if "sudo" in command:
                return "Timed out while waiting for password, sudo not supported yet. Be careful!"
            return "Timeout error"
        if self.process.after == pexpect.EOF:
            return f"Exited with error status: {self.process.exitstatus}"
        if self.strip_newlines:
            return self.process.before.strip()
        return self.process.before
