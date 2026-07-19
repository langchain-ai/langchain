"""Tests for the vendored mustache renderer."""

from langchain_core.utils.mustache import render


def test_dotted_path_through_falsy_intermediate_zero() -> None:
    # `x` is the scalar 0, which has no member `y`, so `x.y` does not resolve
    # and should render empty -- not leak the intermediate value `0`.
    assert render("a{{x.y}}b", {"x": 0}) == "ab"


def test_dotted_path_through_falsy_intermediate_false() -> None:
    assert render("a{{x.y}}b", {"x": False}) == "ab"


def test_dotted_path_final_zero_is_rendered() -> None:
    # When the full path resolves to 0, 0 must still render as "0".
    assert render("a{{x.y}}b", {"x": {"y": 0}}) == "a0b"


def test_dotted_path_final_false_is_rendered() -> None:
    assert render("a{{x.y}}b", {"x": {"y": False}}) == "aFalseb"


def test_top_level_falsy_is_rendered() -> None:
    assert render("a{{x}}b", {"x": 0}) == "a0b"
    assert render("a{{x}}b", {"x": False}) == "aFalseb"
