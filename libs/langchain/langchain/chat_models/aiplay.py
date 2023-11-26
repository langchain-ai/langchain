"""Chat Model Components Derived from LLM/aiplay"""

from langchain.llms.aiplay import AIPlayBaseModel
from langchain.pydantic_v1 import Field

## NOTE: This file should not be ran in isolation as a single-file standalone.
## Please use llms.aiplay instead.
from .base import SimpleChatModel  


class AIPlayChat(AIPlayBaseModel, SimpleChatModel):
    pass

class LlamaChat(AIPlayChat):
    model_name : str = Field(default='llama2_13b', alias='model')

class MistralChat(AIPlayChat):
    model_name : str = Field(default="mistral", alias='model')

class SteerLMChat(AIPlayChat):
    model_name : str = Field(default="gpt_steerlm_8b", alias='model')
    labels = Field(default={
        "creativity": 5,
        "helpfulness": 5,
        "humor": 5,
        "quality": 5
    })

class NemotronQAChat(AIPlayChat):
    model_name : str = Field(default="gpt_qa_8b", alias='model')    
