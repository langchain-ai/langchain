"""Chat Model Components Derived from ChatModel/NVAIPlay"""
from langchain_core.language_models.chat_models import SimpleChatModel

from langchain_nvidia_aiplay import _common as nv_aiplay


class ChatNVAIPlay(nv_aiplay.NVAIPlayBaseModel, SimpleChatModel):
    """NVAIPlay chat model.

    Example:
        .. code-block:: python

            from langchain_nvidia_aiplay import ChatNVAIPlay


            model = ChatNVAIPlay(model="mistral")
            response = model.invoke("Hello")
    """
