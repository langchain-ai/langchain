"""Azure OpenAI chat wrapper."""
from __future__ import annotations

import logging

from langchain.chat_models.openai import ChatOpenAI
from langchain.llms.openai import AzureOpenAIMixin

logger = logging.getLogger(__name__)


class AzureChatOpenAI(ChatOpenAI, AzureOpenAIMixin):
    """Wrapper around Azure OpenAI Chat Completion API. To use this class you
    must have a deployed model on Azure OpenAI. Use `deployment_name` in the
    constructor to refer to the "Model deployment name" in the Azure portal.

    In addition, you should have the ``openai`` python package installed, and the
    following environment variables set or passed in constructor in lower case:
    - ``OPENAI_API_TYPE`` (default: ``azure``)
    - ``OPENAI_API_KEY``
    - ``OPENAI_API_BASE``
    - ``OPENAI_API_VERSION``

    For exmaple, if you have `gpt-35-turbo` deployed, with the deployment name
    `35-turbo-dev`, the constructor should look like:

    .. code-block:: python
        AzureChatOpenAI(
            deployment_name="35-turbo-dev",
            openai_api_version="2023-03-15-preview",
        )

    Be aware the API version may change.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.
    """
