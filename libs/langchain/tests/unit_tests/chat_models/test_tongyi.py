"""
This file is to perform unit tests for Tongyi chat model.
"""

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.chat_models.tongyi import ChatTongyi

