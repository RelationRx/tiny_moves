from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tiny_moves.metrics.utils.rx_data import CELL_TYPES
from tiny_moves.metrics.utils.stats.permutation import calculate_pvalue
from tiny_moves.representations.structured_representations.differentially_expressed_genes import (
    DIFFERENTIALLY_EXPRESSED_DIRECTION,
)


def bin_by_global_expression(expr_df: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    """
    Bin genes by their global mean expression across all cell types.

    Args:
        expr_df: DataFrame of shape (genes by cell types), with numeric expression values.
        n_bins: Number of bins to create.

    Returns:
        dictionary with bin labels for each gene based on global mean expression.

    """
    # Bin genes by global mean expression
    global_expr = cast(pd.Series, expr_df.mean(axis=1))
    bin_labels = pd.Series(pd.qcut(global_expr, q=n_bins, labels=False, duplicates="drop"), index=global_expr.index)

    return bin_labels


def stratified_null_sampling(
    binned_genes: pd.Series,
    focal_genes: Sequence[str],
    n_iter: int,
    rng: np.random.Generator,
) -> NDArray[np.str_]:
    """
    Sample genes from the same expression bins as focal genes, excluding focal genes.

    Args:
        binned_genes: pand series with the index as gene names and values as bin labels.
        bin_labels: Integer bin assignments (same length as gene_array).
        focal_genes: Genes to match bin structure for.
        n_iter: Number of permutations.
        rng: NumPy RNG.

    Returns:
        np.ndarray of shape (n_iter, len(focal_genes)) of sampled gene names.

    """
    focal_bins = binned_genes.loc[focal_genes].astype(int).to_numpy()

    # Create pool of background genes
    non_focal_genes = binned_genes.index.difference(focal_genes)
    non_focal_bins = binned_genes.loc[non_focal_genes].astype(int)

    # Group gene names by bin
    bin_to_pool = non_focal_bins.groupby(non_focal_bins).groups

    # Sample per focal gene from corresponding bin
    result = np.empty((n_iter, len(focal_genes)), dtype=object)
    for j, b in enumerate(focal_bins):
        pool = bin_to_pool.get(b)
        if pool is None or len(pool) == 0:
            raise ValueError(f"No background genes available for bin {b}")
        result[:, j] = rng.choice(pool, size=n_iter, replace=True)

    return result


def celltype_expression_specificity_score(
    expr_df: pd.DataFrame,
    gene_set: Sequence[str],
    celltype: CELL_TYPES,
) -> dict[str, float]:
    """
    Compute a Z-score for specificity of a gene set's expression in a cell type versus all others.

    At the moment this uses ordinal expression categories. In future it would be better to
    calculate this using actual expression values in a formal logfold change set up.

    Args:
        expr_df: DataFrame of shape (genes by cell types), with numeric expression values.
        gene_set: List of genes to test (must be in expr_df.index).
        celltype: Name of the focal cell type (must be in expr_df.columns).

    Returns:
        Dictionary with focal median, mean of other cell types medians, std of other cell type medians,
        and specificity score.

    """
    # Subset matrix
    celltypes = expr_df.columns.drop(celltype)

    gene_expr = expr_df.loc[gene_set]

    # Median expression in focal cell type
    focal_median = gene_expr[celltype].median()

    # median expression across each of the other cell types
    other_medians = gene_expr[celltypes].median(axis=0)  # 1 median per cell type

    # Compute specificity
    mean_other_medians = other_medians.mean()
    std_others = other_medians.std(ddof=1)

    if std_others == 0:
        specificity = np.nan
    else:
        specificity = (focal_median - mean_other_medians) / std_others

    return {
        "focal_median": focal_median,
        "mean_other_celltype_medians": mean_other_medians,
        "std_other_celltype_medians": std_others,
        "specificity_score": specificity,
    }


def celltype_expression_specificity_permutation(
    expr_df: pd.DataFrame,
    gene_set: Sequence[str],
    celltype: CELL_TYPES,
    direction: DIFFERENTIALLY_EXPRESSED_DIRECTION,
    score_fn: Callable[
        [pd.DataFrame, Sequence[str], CELL_TYPES], dict[str, float]
    ] = celltype_expression_specificity_score,
    n_iter: int = 10000,
    n_bins: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Perform a statistical test for gene set specificity in a target cell type.

    Statistical test uses stratified sampling to calculate null distribution of cell type specificty.
    Sampling of null distribution is stratified by global mean expression of the focal genes.

    Args:
        expr_df: DataFrame of shape (genes by cell types), with numeric expression values.
        gene_set: List of genes to test (must be in expr_df.index).
        celltype: Name of the focal cell type (must be in expr_df.columns).
        direction: Direction of differential expression ('UP', 'DOWN', 'UNDEFINED').
        score_fn: Function to compute specificity score (default is celltype_expression_specificity_score).
        n_iter: Number of iterations for null distribution.
        n_bins: Number of bins for stratified sampling based on global mean expression.
        random_state: Seed for random number generator.

    Returns:
        Dictionary with observed score, p-value, and null mean score.

    """
    rng = np.random.default_rng(random_state)
    expr_df_copy: pd.DataFrame = expr_df.copy()

    if celltype not in expr_df_copy.columns:
        raise ValueError(f"Target cell type '{celltype}' not found in expr_df columns.")

    focal_gene_set = list(expr_df_copy.index.intersection(gene_set))
    if not focal_gene_set:
        raise ValueError("None of the input genes are in the expression DataFrame.")

    # Observed score
    obs_score = score_fn(expr_df_copy, focal_gene_set, celltype)["specificity_score"]

    # Null distribution
    # Build bin → background gene pool map
    bin_labels = bin_by_global_expression(expr_df_copy, n_bins=n_bins)

    sampled_gene_matrix = stratified_null_sampling(
        binned_genes=bin_labels,
        focal_genes=focal_gene_set,
        n_iter=n_iter,
        rng=rng,
    )

    null_scores = np.array(
        [score_fn(expr_df, list(sampled_genes), celltype)["specificity_score"] for sampled_genes in sampled_gene_matrix]
    )

    pval = calculate_pvalue(observed=obs_score, null_distribution=null_scores, direction=direction)

    return {"observed_score": obs_score, "p_value": float(pval), "null_mean_score": float(np.nanmean(null_scores))}


def celltype_median_expression_rank(
    expr_df: pd.DataFrame,
    gene_set: Sequence[str],
    celltype: CELL_TYPES,
    direction: DIFFERENTIALLY_EXPRESSED_DIRECTION,
    n_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Calculate the mean expression rank of a gene set in a focal cell type.

    Args:
        expr_df: DataFrame of shape (genes bycell types), with numeric expression values.
        gene_set: List of genes to test (must be in expr_df.index).
        celltype: Name of the focal cell type (must be in expr_df.columns).
        direction: Direction of differential expression ('UP', 'DOWN', 'UNDEFINED').
        n_iter: Number of iterations for null distribution.
        random_state: Seed for random number generator.

    Returns:
        Dictionary with observed mean rank, p-value, and null mean rank.

    """
    rng = np.random.default_rng(random_state)
    sample_size = len(gene_set)
    all_genes = expr_df.index.to_numpy()

    expr_col = expr_df[celltype]

    # Observed mean rank
    valid_genes = expr_df.index.intersection(gene_set)
    if len(valid_genes) == 0:
        raise ValueError("No valid genes from gene_set found in expression DataFrame index.")

    obs_values = expr_col.loc[valid_genes]
    obs_median = obs_values.median()

    # Null distribution of mean ranks from random gene sets
    null_medians = np.empty(n_iter)
    for i in range(n_iter):
        random_genes = rng.choice(all_genes, size=sample_size, replace=False)
        null_medians[i] = expr_col.loc[random_genes].median()

    pval = calculate_pvalue(observed=obs_median, null_distribution=null_medians, direction=direction)
    return {"observed_median_rank": obs_median, "p_value": pval, "null_median_rank": null_medians.mean()}
