from typing import Dict, Type

from langchain_glm.agent_toolkits import AdapterAllTool
from langchain_glm.agent_toolkits.all_tools.code_interpreter_tool import (
    CodeInterpreterAdapterAllTool,
)
from langchain_glm.agent_toolkits.all_tools.drawing_tool import (
    DrawingAdapterAllTool,
)
from langchain_glm.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_glm.agent_toolkits.all_tools.web_browser_tool import (
    WebBrowserAdapterAllTool,
)

TOOL_STRUCT_TYPE_TO_TOOL_CLASS: Dict[AdapterAllToolStructType, Type[AdapterAllTool]] = {
    AdapterAllToolStructType.CODE_INTERPRETER: CodeInterpreterAdapterAllTool,
    AdapterAllToolStructType.DRAWING_TOOL: DrawingAdapterAllTool,
    AdapterAllToolStructType.WEB_BROWSER: WebBrowserAdapterAllTool,
}
