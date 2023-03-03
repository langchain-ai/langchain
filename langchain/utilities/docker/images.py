"""This module defines template images and halpers for common docker images."""

from enum import Enum
from typing import Optional, List, Type, Union
from pydantic import BaseModel, Extra, validator



class BaseImage(BaseModel, extra=Extra.forbid):
    """Base docker image template class."""
    tty: bool = False
    stdin_open: bool = True
    name: str
    tag: Optional[str] = 'latest'
    default_command: Optional[List[str]] = None
    stdin_command: Optional[List[str]] = None
    network: str = 'none'

    def dict(self, *args, **kwargs):
        """Override the dict method to add the image name."""
        d = super().dict(*args, **kwargs)
        del d['name']
        del d['tag']
        d['image'] = self.image_name
        return d

    @property
    def image_name(self) -> str:
        """Image name."""
        return f'{self.name}:{self.tag}'



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
    name: str = 'alpine'
    default_command: List[str] = [ShellTypes.sh.value, '-c']
    stdin_command: List[str] = [ShellTypes.sh.value, '-i']

    @validator('default_command')
    def validate_default_command(cls, value: str) -> str:
        """Validate shell type."""
        val = getattr(ShellTypes, value, None)
        if val:
            return val.value
        return value

    @validator('stdin_command')
    def validate_stdin_command(cls, value: str) -> str:
        """Validate shell type."""
        val = getattr(ShellTypes, value, None)
        if val:
            return val.value
        return value

# example using base image to construct python image
class Python(BaseImage):
    """Python image class.

        The python image needs to be launced using the `python3 -i` command to keep
        stdin open.
    """
    name: str = 'python'
    default_command: List[str] = ['python3', '-c']
    stdin_command: List[str] = ['python3', '-iq']


def get_image_template(image_name: str = 'shell') -> Union[str, Type[BaseImage]]:
    """Helper to get an image template from a string.

    It tries to find a class with the same name as the image name and returns the
    class. If no class is found, it returns the image name.


		.. code-block:: python

            >>> image = get_image_template('python')
            >>> assert type(image) == Python
    """
    import importlib
    import inspect

    classes = inspect.getmembers(importlib.import_module(__name__),
                                 lambda x: inspect.isclass(x) and x.__name__ == image_name.capitalize()
                                 )

    if classes:
        cls = classes[0][1]
        return cls
    else:
        return image_name

