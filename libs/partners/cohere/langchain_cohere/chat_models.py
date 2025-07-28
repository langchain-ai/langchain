"""Chat model for Cohere."""

from typing import Any, Dict, List, Optional, Union

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage
except ImportError:
    # Fallback for when langchain_core is not available
    class BaseChatModel:
        pass
    class BaseMessage:
        pass


class ChatCohere(BaseChatModel):
    """Cohere chat model.
    
    To use this, you need to install the ``langchain-cohere`` package.
    
    .. code-block:: bash
    
        pip install langchain-cohere
    
    Example:
        .. code-block:: python
        
            from langchain_cohere import ChatCohere
            
            chat = ChatCohere(model="command-r-plus")
            messages = [("human", "Hello, how are you?")]
            response = chat.invoke(messages)
    
    """
    
    model: str = "command-r-plus"
    """Model name to use."""
    
    cohere_api_key: Optional[str] = None
    """Cohere API key. If not provided, will read from environment variable COHERE_API_KEY."""
    
    temperature: float = 0.0
    """Temperature for sampling."""
    
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate chat completion."""
        raise NotImplementedError(
            "This is a placeholder class. Install langchain-cohere to use: pip install langchain-cohere"
        )
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "cohere-chat"