import os
from collections.abc import Generator
from unittest.mock import patch

import pytest

from langchain.evaluation.hallucination.detector import HallucinationDetector

# -----------------------------
# Integration Tests (Real HF model)
# -----------------------------
skip_if_no_hf = pytest.mark.skipif(
    "HF_TOKEN" not in os.environ,
    reason="Hugging Face token not available"
)


@pytest.fixture(scope="module")
@skip_if_no_hf
@pytest.mark.requires("integration")
def detector_real() -> HallucinationDetector:
    """Runs only if Hugging Face token is available."""
    return HallucinationDetector(model_name="facebook/bart-large-mnli")


@skip_if_no_hf
@pytest.mark.requires("integration")
def test_extract_claims_integration(detector_real: HallucinationDetector) -> None:
    text = (
    "Barack Obama was the 44th President of the United States. "
    "He was born in Kenya."
    )
    claims = detector_real.extract_claims(text)
    # Check structure and basic logic
    assert isinstance(claims, list)
    assert len(claims) == 2
    # Ensure at least one claim matches expected
    assert any("Barack Obama was the 44th President" in c for c in claims)


@skip_if_no_hf
@pytest.mark.requires("integration")
def test_compute_hallucination_rate_integration(
    detector_real: HallucinationDetector,
) -> None:
    text = (
    "Barack Obama was the 44th President of the United States. "
    "He was born in Kenya."
    )
    evidence = [
        (
        "Barack Obama served as the 44th President of the United States "
        "from 2009 to 2017."
        ),
        "Barack Obama was born in Hawaii, not Kenya.",
    ]
    result = detector_real.compute_hallucination_rate(text, evidence)

    # Validate structure
    for key in ["total_claims", "unsupported_claims", "hallucination_rate"]:
        assert key in result

    total = result["total_claims"]
    unsupported = result["unsupported_claims"]
    hallucination_rate = result["hallucination_rate"]

    assert total == 2
    assert 0 <= unsupported <= total
    assert abs(hallucination_rate - unsupported / total) < 1e-6
    assert 0 <= hallucination_rate <= 1


# -----------------------------
# Unit Tests (Mocked)
# -----------------------------
@pytest.fixture(scope="module")
def detector_mock() -> Generator[HallucinationDetector, None, None]:
    """Mock pipeline to make unit tests deterministic."""
    with patch("langchain.evaluation.hallucination.detector.pipeline") as mock_pipeline:
        # Mock NLI behavior
        mock_pipeline.return_value = lambda text: [
            {"label": "ENTAILMENT", "score": 0.9}
            if "President" in text
            else {"label": "CONTRADICTION", "score": 0.9}
        ]
        detector = HallucinationDetector(model_name="any")  # Model not loaded
        yield detector


def test_extract_claims_mock(detector_mock: HallucinationDetector) -> None:
    text = (
        "Barack Obama was the 44th President of the United States. "
        "He was born in Kenya."
    )
    claims = detector_mock.extract_claims(text)
    assert isinstance(claims, list)
    assert len(claims) == 2


def test_verify_claim_supported_mock(detector_mock: HallucinationDetector) -> None:
    claim = "Barack Obama was the 44th President of the United States"
    evidence = (
        "Barack Obama served as the 44th President of the United States "
        "from 2009 to 2017."
    )
    assert detector_mock.verify_claim(claim, evidence) is True


def test_verify_claim_unsupported_mock(detector_mock: HallucinationDetector) -> None:
    claim = "Barack Obama was born in Kenya"
    evidence = "Barack Obama was born in Hawaii, not Kenya."
    assert detector_mock.verify_claim(claim, evidence) is False


def test_compute_hallucination_rate_mock(detector_mock: HallucinationDetector) -> None:
    text = (
        "Barack Obama was the 44th President of the United States. "
        "He was born in Kenya."
    )
    evidence = [
        (
        "Barack Obama served as the 44th President of the United States "
        "from 2009 to 2017.",
        ),
        "Barack Obama was born in Hawaii, not Kenya.",
    ]
    result = detector_mock.compute_hallucination_rate(text, evidence)
    # Validate structure and logical consistency
    for key in ["total_claims", "unsupported_claims", "hallucination_rate"]:
        assert key in result
    assert result["total_claims"] == 2
    assert 0 <= result["unsupported_claims"] <= 2
    assert (
        abs(
            result["hallucination_rate"]
            - result["unsupported_claims"] / result["total_claims"]
        )
        < 1e-6
    )
