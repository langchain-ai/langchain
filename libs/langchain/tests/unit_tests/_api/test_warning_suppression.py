import importlib
import sys
import warnings
from types import ModuleType

import pytest
from langchain_core._api import (
    LangChainDeprecationWarning,
    suppress_langchain_deprecation_warning,
)


@pytest.mark.parametrize("namespace", ["chat_models", "llms", "tools"])
def test_community_import_warning_respects_suppression(
    namespace: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"langchain_classic.{namespace}")
    community = ModuleType("langchain_community")
    target = ModuleType(f"langchain_community.{namespace}")
    imported = object()
    monkeypatch.setattr(target, "Example", imported, raising=False)
    setattr(community, namespace, target)
    monkeypatch.setitem(sys.modules, "langchain_community", community)
    monkeypatch.setattr(module, "is_interactive_env", lambda: False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", LangChainDeprecationWarning)
        with suppress_langchain_deprecation_warning():
            assert module.Example is imported
            with suppress_langchain_deprecation_warning():
                assert module.Example is imported
            assert module.Example is imported
        with pytest.raises(
            LangChainDeprecationWarning, match=r"Importing .* deprecated"
        ):
            _ = module.Example

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LangChainDeprecationWarning)
        assert module.Example is imported
        assert module.Example is imported
    assert len(caught) == 2
    assert all(w.category is LangChainDeprecationWarning for w in caught)
    assert all(w.filename == __file__ for w in caught)
    assert all(
        f"from langchain_community.{namespace} import Example" in str(w.message)
        for w in caught
    )

    monkeypatch.setattr(module, "is_interactive_env", lambda: True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LangChainDeprecationWarning)
        assert module.Example is imported
