from typing import cast

import pandas as pd


def beta_binomial_posterior_mean(
    df: pd.DataFrame,
    hit_col: str,
    effect_size_col: str | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute the Beta-Binomial posterior mean.

    The posterior mean is given by the formula:
        (k + alpha) / (n + alpha + beta)

    k = sum of hits (or hit * effect_size if provided)
    n = number of trials (or count of rows)

    Args:
        df: DataFrame of observations.
        hit_col: column name for successes (0 or 1).
        effect_size_col: optional column for weights (must be normalized ≤1).
        alpha: prior alpha
        beta:  prior beta

    Returns:
        posterior mean as float

    """
    hits = df[hit_col].astype(float)
    n = hits.count()

    if effect_size_col:
        eff = cast(pd.Series, df[effect_size_col].clip(lower=0))
        if eff.max() > 1.0:
            raise ValueError(f"{effect_size_col} column is not normalized")
        k = (hits * eff).sum()
    else:
        k = hits.sum()

    return float((k + alpha) / (n + alpha + beta))
