from langchain.pydantic_v1 import BaseModel


class TrainResult(BaseModel):
    loss: float
