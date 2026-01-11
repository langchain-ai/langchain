from __future__ import annotations

import pytest

from langchain_core.utils.math import beta_binomial_posterior_mean


def test_beta_binomial_posterior_mean_with_uniform_prior() -> None:
    result = beta_binomial_posterior_mean(3, 1)

    assert result == pytest.approx(2 / 3)


def test_beta_binomial_posterior_mean_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        beta_binomial_posterior_mean(-1, 0)


def test_beta_binomial_posterior_mean_rejects_invalid_prior() -> None:
    with pytest.raises(ValueError, match="positive"):
        beta_binomial_posterior_mean(1, 1, alpha=0.0)
