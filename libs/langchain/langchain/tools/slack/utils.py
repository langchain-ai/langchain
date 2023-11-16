"""Slack tool utils."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


def login() -> WebClient:
    """Authenticate using the Slack API."""
    try:
        from slack_sdk import WebClient
    except ImportError as e:
        raise ImportError(
            "Cannot import slack_sdk. Please install the package with `pip install slack_sdk`."
        ) from e

    if "SLACK_BOT_TOKEN" in os.environ:
        token = os.environ["SLACK_BOT_TOKEN"]
        client = WebClient(token=token)
        logger.info("slack login success")
        return client
    elif "SLACK_USER_TOKEN" in os.environ:
        token = os.environ["SLACK_USER_TOKEN"]
        client = WebClient(token=token)
        logger.info("slack login success")
        return client
    else:
        logger.error(
            "Error: The SLACK_BOT_TOKEN or SLACK_USER_TOKEN environment variable have not "
            "been set."
        )


# def login() -> WebClient:
#     try:
#         from slack_sdk import WebClient
#     except ImportError as e:
#         raise ImportError(
#             "Cannot import slack_sdk. Please install the package with `pip install slack_sdk`."
#         ) from e

#     client = WebClient(token="xoxp-6040211441731-6052945665441-6048335428100-5909cdff7e6e11a55541834dc69dd9b6")
#     return client


def authenticate() -> WebClient:
    """Authenticate using the Microsoft Grah API"""
    try:
        logger.error("slack auth start")
        from slack_sdk import WebClient

        client = WebClient(token=os.environ.get("SLACK_USR_TOKEN"))
        logger.error("get client")
        return client
    except ImportError as e:
        # raise ImportError(
        #     "Cannot import slack_sdk. Please install the package with `pip install slack_sdk`."
        # ) from e
        logger.error("slack auth failed")
