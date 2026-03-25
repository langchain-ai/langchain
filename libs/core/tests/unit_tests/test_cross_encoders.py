"""Tests for cross_encoders module."""

import pytest

from langchain_core.cross_encoders import BaseCrossEncoder


class MockCrossEncoder(BaseCrossEncoder):
    """Mock implementation of BaseCrossEncoder for testing."""

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Return mock scores based on text length similarity."""
        scores = []
        for text1, text2 in text_pairs:
            # Simple mock scoring based on length similarity
            len1, len2 = len(text1), len(text2)
            max_len = max(len1, len2) if max(len1, len2) > 0 else 1
            similarity = 1.0 - abs(len1 - len2) / max_len
            scores.append(similarity)
        return scores


def test_base_cross_encoder_is_abstract():
    """Test that BaseCrossEncoder cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        BaseCrossEncoder()


def test_mock_cross_encoder_score():
    """Test that MockCrossEncoder correctly implements score method."""
    encoder = MockCrossEncoder()

    # Test with single pair
    pairs = [("hello world", "hello there")]
    scores = encoder.score(pairs)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0

    # Test with multiple pairs
    pairs = [
        ("short", "short"),
        ("long text here", "different long text"),
        ("a", "very long text indeed"),
    ]
    scores = encoder.score(pairs)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Identical texts should have perfect similarity
    assert scores[0] == 1.0

    # Very different lengths should have lower similarity
    assert scores[2] < scores[0]


def test_cross_encoder_empty_pairs():
    """Test cross encoder with empty pairs list."""
    encoder = MockCrossEncoder()
    scores = encoder.score([])
    assert scores == []
