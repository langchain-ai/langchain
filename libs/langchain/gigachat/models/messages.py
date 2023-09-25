from langchain.pydantic_v1 import BaseModel

from .messages_role import MessagesRole


class Messages(BaseModel):
    role: MessagesRole
    content: str

    class Config:
        use_enum_values = True
