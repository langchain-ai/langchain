"""**Chat Loaders** load chat messages from common communications platforms.

Load chat messages from various
communications platforms such as Facebook Messenger, Telegram, and
WhatsApp. The loaded chat messages can be used for fine-tuning models.

**Class hierarchy:**

.. code-block::

    BaseChatLoader --> <name>ChatLoader  # Examples: WhatsAppChatLoader, IMessageChatLoader

**Main helpers:**

.. code-block::

    ChatSession

"""  # noqa: E501

import importlib
from typing import Any

_module_lookup = {
    "BaseChatLoader": "langchain_community.chat_loaders.base",
    "FolderFacebookMessengerChatLoader": "langchain_community.chat_loaders.facebook_messenger",  # noqa: E501
    "GMailLoader": "langchain_community.chat_loaders.gmail",
    "IMessageChatLoader": "langchain_community.chat_loaders.imessage",
    "LangSmithDatasetChatLoader": "langchain_community.chat_loaders.langsmith",
    "LangSmithRunChatLoader": "langchain_community.chat_loaders.langsmith",
    "SingleFileFacebookMessengerChatLoader": "langchain_community.chat_loaders.facebook_messenger",  # noqa: E501
    "SlackChatLoader": "langchain_community.chat_loaders.slack",
    "TelegramChatLoader": "langchain_community.chat_loaders.telegram",
    "WhatsAppChatLoader": "langchain_community.chat_loaders.whatsapp",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
