from typing import Any

import pytest
from langchain_core._api import LangChainDeprecationWarning

from langchain_classic import hub


class FakeLangSmithClient:
    def pull_prompt(self, owner_repo_commit: str, **_: Any) -> str:
        return owner_repo_commit

    def push_prompt(self, repo_full_name: str, **_: Any) -> str:
        return repo_full_name


def test_pull_warns_about_sdk_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, "_get_client", lambda **_: FakeLangSmithClient())

    with pytest.warns(
        LangChainDeprecationWarning,
        match="langsmith.Client.pull_prompt",
    ):
        assert hub.pull("owner/prompt") == "owner/prompt"


def test_push_warns_about_sdk_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, "_get_client", lambda **_: FakeLangSmithClient())

    with pytest.warns(
        LangChainDeprecationWarning,
        match="langsmith.Client.push_prompt",
    ):
        assert hub.push("owner/prompt", object={}) == "owner/prompt"
