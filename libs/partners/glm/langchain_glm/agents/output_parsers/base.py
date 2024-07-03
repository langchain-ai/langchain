from typing import Any, Dict, Optional

from zhipuai.core import BaseModel


class AllToolsMessageToolCall(BaseModel):
    name: Optional[str]
    args: Optional[Dict[str, Any]]
    id: Optional[str]


class AllToolsMessageToolCallChunk(BaseModel):
    name: Optional[str]
    args: Optional[Dict[str, Any]]
    id: Optional[str]
    index: Optional[int]
