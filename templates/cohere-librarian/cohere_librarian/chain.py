# from .router import branch
# # from .library_info import chain
# from .blurb_matcher import book_rec_chain
from langchain.pydantic_v1 import BaseModel

from .router import branched_chain


class ChainInput(BaseModel):
    message: str


chain = branched_chain.with_types(input_type=ChainInput)
