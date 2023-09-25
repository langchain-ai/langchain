from langchain.pydantic_v1 import BaseModel


class MessagesRes(BaseModel):
    role: str
    content: str
