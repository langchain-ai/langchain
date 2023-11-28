"""Chat Model Components Derived from LLM/NVAIPlay"""

from langchain.llms.nv_aiplay import NVAIPlayBaseModel
from langchain.pydantic_v1 import Field

## NOTE: This file should not be ran in isolation as a single-file standalone.
## Please use llms.nv_aiplay instead.
from .base import SimpleChatModel  


class NVAIPlayChat(NVAIPlayBaseModel, SimpleChatModel):
    pass

class LlamaChat(NVAIPlayChat):
    model_name : str = Field("llama2_13b", alias='model')

class MistralChat(NVAIPlayChat):
    model_name : str = Field("mistral", alias='model')

class SteerLMChat(NVAIPlayChat):
    model_name : str = Field("gpt_steerlm_8b", alias='model')
    labels = Field({
        "creativity": 5,
        "helpfulness": 5,
        "humor": 5,
        "quality": 5
    })

class NemotronQAChat(NVAIPlayChat):
    model_name : str = Field("gpt_qa_8b", alias='model')    
