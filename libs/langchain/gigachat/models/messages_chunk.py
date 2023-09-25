from langchain.pydantic_v1 import BaseModel


class MessagesChunk(BaseModel):
    content: str
