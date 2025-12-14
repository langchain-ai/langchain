"""Maritaca AI integration for LangChain.

Maritaca AI provides Brazilian Portuguese-optimized language models.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

from langchain_maritaca.chat_models import ChatMaritaca
from langchain_maritaca.version import __version__

__all__ = ["ChatMaritaca", "__version__"]
