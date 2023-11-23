"""Chat Model Components Derived from LLM/aiplay"""

from langchain.chat_models.base import SimpleChatModel
from langchain.llms.aiplay import AIPlayBaseModel, NeVAClient
from langchain.pydantic_v1 import Field
from typing import Any

## This file cannot be run in isolation as a single-file standalone.
## Please start with llms.aiplay and uncomment chat variants as seen below.
from .base import SimpleChatModel  

################################################################################

# class AIPlayLLM(AIPlayBaseModel, LLM):
#     pass

class AIPlayChat(AIPlayBaseModel, SimpleChatModel):
    pass

################################################################################

# class LlamaLLM(AIPlayLLM):
#     model_name = Field(default="llama-13B-code")

# class MistralLLM(AIPlayLLM):
#     model_name = Field(default="mistral-7B-inst")

class LlamaChat(AIPlayChat):
    model_name = Field(default='llama-2-13B-chat')

class MistralChat(AIPlayChat):
    model_name = Field(default="mistral-7B-inst")

################################################################################

# class NevaLLM(AIPlayBaseModel, LLM):
#     client: Any = Field(NeVAClient)
#     model_name = Field(default="neva-22b")

# class FuyuLLM(AIPlayBaseModel, LLM):
#     client: Any = Field(NeVAClient)
#     model_name = Field(default="fuyu-8b")

class NevaChat(AIPlayBaseModel, SimpleChatModel):
    client: Any = Field(NeVAClient)
    model_name = Field(default="neva-22b")

class FuyuChat(AIPlayBaseModel, SimpleChatModel):
    client: Any = Field(NeVAClient)
    model_name = Field(default="fuyu-8b")

