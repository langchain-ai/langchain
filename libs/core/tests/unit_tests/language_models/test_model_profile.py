"""Tests for model profile types and utilities."""

import warnings
from typing import Any
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field

from langchain_core.language_models.model_profile import (
    ModelProfile,
    _warn_unknown_profile_keys,
)


class TestModelProfileExtraAllow:
    """Verify extra='allow' on ModelProfile TypedDict."""

    def test_accepts_declared_keys(self) -> None:
        profile: ModelProfile = {"max_input_tokens": 100, "tool_calling": True}
        assert profile["max_input_tokens"] == 100

    def test_extra_keys_accepted_via_typed_dict(self) -> None:
        """ModelProfile TypedDict allows extra keys at construction."""
        profile = ModelProfile(
            max_input_tokens=100,
            unknown_future_field="value",  # type: ignore[typeddict-unknown-key]
        )
        assert profile["unknown_future_field"] == "value"  # type: ignore[typeddict-item]

    def test_extra_keys_survive_pydantic_validation(self) -> None:
        """Extra keys pass through even when parent model forbids extras."""

        class StrictModel(BaseModel):
            model_config = ConfigDict(extra="forbid")
            profile: ModelProfile | None = Field(default=None)

        m = StrictModel(
            profile={
                "max_input_tokens": 100,
                "unknown_future_field": True,
            }
        )
        assert m.profile is not None
        assert m.profile.get("unknown_future_field") is True


class TestWarnUnknownProfileKeys:
    """Tests for _warn_unknown_profile_keys."""

    def test_warns_on_extra_keys(self) -> None:
        profile: dict[str, Any] = {
            "max_input_tokens": 100,
            "future_field": True,
            "another": "val",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unknown_profile_keys(profile)  # type: ignore[arg-type]

        assert len(w) == 1
        assert "another" in str(w[0].message)
        assert "future_field" in str(w[0].message)
        assert "upgrading langchain-core" in str(w[0].message)

    def test_silent_on_declared_keys_only(self) -> None:
        profile: ModelProfile = {"max_input_tokens": 100, "tool_calling": True}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unknown_profile_keys(profile)

        assert len(w) == 0

    def test_silent_on_empty_profile(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unknown_profile_keys({})

        assert len(w) == 0

    def test_survives_get_type_hints_failure(self) -> None:
        """Falls back to silent skip on TypeError from get_type_hints."""
        profile: dict[str, Any] = {"max_input_tokens": 100, "extra": True}
        with patch(
            "langchain_core.language_models.model_profile.get_type_hints",
            side_effect=TypeError("broken"),
        ):
            _warn_unknown_profile_keys(profile)  # type: ignore[arg-type]
