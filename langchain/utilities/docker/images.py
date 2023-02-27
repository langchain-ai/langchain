"""Optimized parameters for commonly used docker images that can be used by
the docker wrapper utility to attach to."""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Extra, validator


class BaseImage(BaseModel, extra=Extra.forbid):
    """Base docker image class."""
    tty: bool = False
    stdin_open: bool = True
    image: str
    default_command: Optional[List[str]] = None

class ShellTypes(str, Enum):
    """Enum class for shell types."""
    bash = '/bin/bash'
    sh = '/bin/sh'
    zsh = '/bin/zsh'


class Shell(BaseImage):
    """Shell image focused on running shell commands.

    A shell image can be crated by passing a shell alias such as `sh` or `bash`
    or by passing the full path to the shell binary.
    """
    image: str = 'alpine'
    shell: str = ShellTypes.bash.value

    @validator('shell')
    def validate_shell(cls, value: str) -> str:
        """Validate shell type."""
        val = getattr(ShellTypes, value, None)
        if val:
            return val.value
        # elif value in [v.value for v in list(ShellTypes.__members__.values())]:
        #     print(f"docker: overriding shell binary to: {value}")
        return value

# example using base image to construct python image
class Python(BaseImage):
    """Python image class.

        The python image needs to be launced using the `python3 -i` command to keep
        stdin open.
    """
    image: str = 'python'
    default_command: List[str] = ['python3', '-i']

    def __setattr__(self, name, value):
        if name == 'default_command':
            raise AttributeError(f'running this image with {self.default_command}' 
                                 ' is necessary to keep stdin open.')

        super().__setattr__(name, value)
