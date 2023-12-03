"""
This file is to perform unit tests for Tongyi llm.
"""

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.tongyi import Tongyi