from langchain.pydantic_v1 import BaseModel

from .router import branched_chain


class ChainInput(BaseModel):
    message: str


chain = branched_chain.with_types(input_type=ChainInput)
