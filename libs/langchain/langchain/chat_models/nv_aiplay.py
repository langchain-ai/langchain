"""Chat Model Components Derived from ChatModel/NVAIPlay"""

from langchain.llms import nv_aiplay

from .base import SimpleChatModel


class NVAIPlayChat(nv_aiplay.NVAIPlayBaseModel, SimpleChatModel):
    pass


class GeneralChat(nv_aiplay.GeneralBase, SimpleChatModel):
    pass


class CodeChat(nv_aiplay.CodeBase, SimpleChatModel):
    pass


class InstructChat(nv_aiplay.InstructBase, SimpleChatModel):
    pass


class SteerChat(nv_aiplay.SteerBase, SimpleChatModel):
    pass


class ContextChat(nv_aiplay.ContextBase, SimpleChatModel):
    pass


class ImageChat(nv_aiplay.ImageBase, SimpleChatModel):
    pass
