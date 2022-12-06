import subprocess
from typing import Dict, List, Union

class BashProcess:
    """Executes bash commands and returns the output."""

    def __init__(self, strip_newlines: bool = False):
        self.strip_newlines = strip_newlines


    def run(self, commands: List[str]) -> Dict[str, Union[bool, list[str]]]:
        outputs = []
        for command in commands:
            try:
                output = subprocess.check_output(command, shell=True).decode()
                if self.strip_newlines:
                    output = output.strip()
                outputs.append(output)
            except subprocess.CalledProcessError as error:
                outputs.append(str(error))
                return {"success": False, "outputs": outputs}

        return {"success": True, "outputs": outputs}
