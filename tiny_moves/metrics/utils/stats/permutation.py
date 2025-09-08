import numpy as np
from numpy.typing import NDArray

from tiny_moves.representations.structured_representations.differentially_expressed_genes import (
    DIFFERENTIALLY_EXPRESSED_DIRECTION,
)


def calculate_pvalue(
    observed: float, null_distribution: NDArray[np.float64], direction: DIFFERENTIALLY_EXPRESSED_DIRECTION
) -> float:
    """
    Calculate the p-value for the observed statistic against a null distribution.

    Args:
        observed: The observed statistic.
        null_distribution: The null distribution of statistics.
        direction: The direction of differential expression (UP, DOWN, UNDEFINED).

    Returns:
        float: The p-value.

    """
    # how often is the null case lower than the observed
    p_low = float(np.nanmean(null_distribution <= observed))

    # how often is the null case higher than observed.
    p_high = float(np.nanmean(null_distribution >= observed))

    # is expression either higher or lower (two tailed test)
    p_two = min(1.0, 2.0 * min(p_high, p_low))

    if direction == "UP":
        return p_high

    elif direction == "DOWN":
        return p_low

    elif direction == "UNDEFINED":
        return p_two
