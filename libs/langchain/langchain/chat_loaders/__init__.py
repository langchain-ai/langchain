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

from langchain.chat_loaders.base import BaseChatLoader, ChatSession
from langchain.chat_loaders.facebook_messenger import (
    FolderFacebookMessengerChatLoader,
    SingleFileFacebookMessengerChatLoader,
)
from langchain.chat_loaders.gmail import GMailLoader
from langchain.chat_loaders.imessage import IMessageChatLoader
from langchain.chat_loaders.langsmith import (
    LangSmithDatasetChatLoader,
    LangSmithRunChatLoader,
)
from langchain.chat_loaders.slack import SlackChatLoader
from langchain.chat_loaders.telegram import TelegramChatLoader
from langchain.chat_loaders.utils import (
    map_ai_messages,
    map_ai_messages_in_session,
    merge_chat_runs,
    merge_chat_runs_in_session,
)
from langchain.chat_loaders.whatsapp import WhatsAppChatLoader

__all__ = [
    "BaseChatLoader",
    "ChatSession",
    "FolderFacebookMessengerChatLoader",
    "GMailLoader",
    "IMessageChatLoader",
    "LangSmithDatasetChatLoader",
    "LangSmithRunChatLoader",
    "merge_chat_runs",
    "merge_chat_runs_in_session",
    "map_ai_messages",
    "map_ai_messages_in_session",
    "SingleFileFacebookMessengerChatLoader",
    "SlackChatLoader",
    "TelegramChatLoader",
    "WhatsAppChatLoader",
]
