"""IndexStructType class."""

from enum import Enum


class AdapterAllToolStructType(str, Enum):
    """

    Attributes:
        DICT ("dict"):

    """

    # TODO: refactor so these are properties on the base class

    CODE_INTERPRETER = "code_interpreter"
    DRAWING_TOOL = "drawing_tool"
    WEB_BROWSER = "web_browser"
