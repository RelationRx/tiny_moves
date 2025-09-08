from collections.abc import Callable
from typing import Union

import pandas as pd

# Type alias for prior configurations: either a fixed (alpha, beta) tuple or a fn that computes them.
PriorType = Union[
    tuple[float, float],
    Callable[[pd.DataFrame, str], tuple[float, float]],
]


def observed_prior(df: pd.DataFrame, hit_col: str, prior_n: int = 10) -> tuple[float, float]:
    """
    Estimate a Beta prior from observed hit rate with pseudo-count mass.

    Args:
        df: DataFrame of observations.
        hit_col: column name for successes (0/1 or float).
        prior_n: total pseudo-count (alpha+beta).

    Returns:
        (alpha, beta) = (p̂ * prior_n, (1-p̂)*prior_n)

    """
    hits = df[hit_col].astype(float)
    p_hat = hits.mean()
    alpha = p_hat * prior_n
    beta = (1 - p_hat) * prior_n
    return alpha, beta


# Built-in priors:
ALPHA_BETA_DEFAULTS: dict[str, PriorType] = {
    "uniform": (1.0, 1.0),
    "none": (0.0, 0.0),
    "observed": observed_prior,
}


def get_prior(name: str, df: pd.DataFrame | None = None, hit_col: str = "hit") -> tuple[float, float]:
    """
    Retrieve (alpha, beta) for a given prior name.

    Args:
        name: key in ALPHA_BETA_DEFAULTS
        df: DataFrame for data-derived priors
        hit_col: column name for hits

    Returns:
        (alpha, beta)

    """
    try:
        prior = ALPHA_BETA_DEFAULTS[name]
    except KeyError as e:
        raise KeyError(f"Unknown prior '{name}'") from e

    if callable(prior):
        if df is None:
            raise ValueError("DataFrame must be provided for data-derived prior")
        return prior(df, hit_col)
    return prior
