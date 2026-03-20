"""Tests for model profile types and utilities."""

import warnings
from typing import Any, ClassVar, get_type_hints
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

from langchain_core.language_models.model_profile import (
    ModelProfile,
    _warn_unknown_profile_keys,
)


def _profile_with_extra(**extra: Any) -> ModelProfile:
    """Build a ModelProfile with extra keys (bypasses static type checking)."""
    base: dict[str, Any] = {"max_input_tokens": 100}
    base.update(extra)
    return base  # type: ignore[return-value]


class TestModelProfileExtraAllow:
    """Verify extra='allow' on ModelProfile TypedDict."""

    def test_accepts_declared_keys(self) -> None:
        profile: ModelProfile = {"max_input_tokens": 100, "tool_calling": True}
        assert profile["max_input_tokens"] == 100

    def test_accepts_extra_keys_at_runtime(self) -> None:
        profile = _profile_with_extra(unknown_future_field="value")
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
        profile = _profile_with_extra(future_field=True, another="val")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unknown_profile_keys(profile)

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
        profile: ModelProfile = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unknown_profile_keys(profile)

        assert len(w) == 0

    def test_survives_get_type_hints_failure(self) -> None:
        """Must never crash — falls back to silent skip."""
        profile = _profile_with_extra(extra=True)
        with patch(
            "langchain_core.language_models.model_profile.get_type_hints",
            side_effect=TypeError("broken"),
        ):
            # Should not raise
            _warn_unknown_profile_keys(profile)

    def test_all_current_declared_fields_recognized(self) -> None:
        """Sanity check: all declared fields are recognized as declared."""
        hints = get_type_hints(ModelProfile)
        profile: dict[str, Any] = {}
        for key, typ in hints.items():
            if typ is bool:
                profile[key] = True
            elif typ is int:
                profile[key] = 100
            elif typ is str:
                profile[key] = "test"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_unknown_profile_keys(profile)  # type: ignore[arg-type]

        assert len(w) == 0


class TestModelProfileFields:
    """Verify expected fields exist on ModelProfile."""

    _declared: ClassVar[set[str]] = set(get_type_hints(ModelProfile).keys())

    @pytest.mark.parametrize(
        "field",
        [
            "name",
            "status",
            "release_date",
            "last_updated",
            "open_weights",
            "max_input_tokens",
            "max_output_tokens",
            "text_inputs",
            "image_inputs",
            "audio_inputs",
            "video_inputs",
            "text_outputs",
            "image_outputs",
            "audio_outputs",
            "video_outputs",
            "tool_calling",
            "tool_choice",
            "structured_output",
            "attachment",
            "temperature",
            "image_url_inputs",
            "image_tool_message",
            "pdf_tool_message",
            "pdf_inputs",
            "reasoning_output",
        ],
    )
    def test_field_declared(self, field: str) -> None:
        assert field in self._declared
