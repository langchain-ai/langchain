"""Math helpers for probability estimates."""

from __future__ import annotations


def beta_binomial_posterior_mean(
    successes: int,
    failures: int,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Compute the posterior mean for a Beta-Binomial model.

    Args:
        successes: Number of observed successes.
        failures: Number of observed failures.
        alpha: Prior alpha parameter.
        beta: Prior beta parameter.

    Returns:
        The posterior mean probability of success.

    Raises:
        ValueError: If any inputs are negative or the posterior is invalid.
    """
    if successes < 0 or failures < 0:
        msg = "successes and failures must be non-negative"
        raise ValueError(msg)
    if alpha <= 0.0 or beta <= 0.0:
        msg = "alpha and beta must be positive"
        raise ValueError(msg)

    posterior_alpha = alpha + successes
    posterior_beta = beta + failures
    denominator = posterior_alpha + posterior_beta
    if denominator <= 0.0:
        msg = "posterior parameters must sum to a positive value"
        raise ValueError(msg)

    return posterior_alpha / denominator
