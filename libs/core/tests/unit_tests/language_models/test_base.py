import builtins
import importlib
from collections.abc import Callable
from typing import Any

import pytest

import langchain_core.language_models.base as language_models_base


def _block_transformers_import(
    real_import: Callable[..., Any],
    *,
    error_factory: Callable[[], Exception],
) -> Callable[..., Any]:
    def _import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "transformers":
            raise error_factory()
        return real_import(name, globals_, locals_, fromlist, level)

    return _import


def test_base_module_does_not_eagerly_import_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    with monkeypatch.context() as m:
        m.setattr(
            builtins,
            "__import__",
            _block_transformers_import(
                real_import,
                error_factory=lambda: AssertionError(
                    "base module should not eagerly import transformers"
                ),
            ),
        )
        reloaded_base = importlib.reload(language_models_base)
        assert reloaded_base._HAS_TRANSFORMERS is None

    importlib.reload(language_models_base)


def test_get_tokenizer_raises_helpful_error_without_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    with monkeypatch.context() as m:
        m.setattr(
            builtins,
            "__import__",
            _block_transformers_import(
                real_import,
                error_factory=lambda: ImportError(
                    "No module named 'transformers'"
                ),
            ),
        )
        reloaded_base = importlib.reload(language_models_base)
        reloaded_base.get_tokenizer.cache_clear()
        reloaded_base._get_gpt2_tokenizer_fast.cache_clear()

        with pytest.raises(
            ImportError,
            match="Could not import transformers python package",
        ):
            reloaded_base.get_tokenizer()

    importlib.reload(language_models_base)
