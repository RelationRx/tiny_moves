from typing import Any

import pandas as pd

from tiny_moves.metrics.pairwise_hypothesis_metrics.template import (
    PairwiseHypothesisMetric,
)


def compute_pairwise_hypothesis_metrics_on_objects(
    objs: list[Any],
    metrics: list[PairwiseHypothesisMetric[Any, Any]],
    index_labels: list[Any] | None = None,
    index_name: str = "pair",
) -> pd.DataFrame:
    """
    Compute pairwise-hypothesis metrics on consecutive objects in a trajectory.

    This function takes a list of hypothesis-like objects and a list of PairwiseHypothesisMetric
    instances. It computes each metric between each object and its next neighbor in the list,
    returning a DataFrame with one row per consecutive pair.

    By default, it compares objs[i] (reference) to objs[i+1] (candidate) for i in [0 .. len(objs)-2].
    You can optionally supply `index_labels` for the pairs; its length must be len(objs)-1.

    The returned DataFrame will include:
      • 'reference_obj': the object at step i
      • 'candidate_obj': the object at step i+1
      • one column per metric, named by the metric's class name

    Args:
        objs: A list of hypothesis-like objects in trajectory order.
        metrics: List of pairwise metrics to compute.
        index_labels: Optional labels for each pair. If provided,
            its length must equal len(objs) - 1. If None, a default RangeIndex is used.
        index_name: Name to assign to the DataFrame Index. Defaults to 'pair'.

    Returns:
        pd.DataFrame: A DataFrame with index labeled by `index_labels` or a RangeIndex,
                      containing 'reference_obj', 'candidate_obj', and one column per metric.

    """
    # If fewer than two objects, there are no consecutive pairs
    if len(objs) < 2:  # noqa: PLR2004
        # Build an empty DataFrame with the expected columns
        return pd.DataFrame(columns=pd.Index(["reference_obj", "candidate_obj"] + [type(m).__name__ for m in metrics]))

    records = []
    # Loop over consecutive pairs: (objs[0], objs[1]), (objs[1], objs[2]), …
    for i in range(len(objs) - 1):
        ref = objs[i]
        cand = objs[i + 1]
        row: dict[str, Any] = {"reference_obj": ref, "candidate_obj": cand}
        for metric in metrics:
            metric_name = type(metric).__name__
            row[metric_name] = metric.compute(ref, cand)
        records.append(row)

    # Determine index (either user-provided or default integer range)
    num_pairs = len(records)
    if index_labels is not None:
        if len(index_labels) != num_pairs:
            raise ValueError(f"Length of index_labels ({len(index_labels)}) must match number of pairs ({num_pairs}).")
        idx = pd.Index(index_labels, name=index_name)
    else:
        idx = pd.Index(range(num_pairs), name=index_name)

    df = pd.DataFrame.from_records(records, index=idx)
    return df


def compute_all_pairwise_metric_values(
    objs: list[Any],
    metric: PairwiseHypothesisMetric[Any, Any],
    index_labels: list[Any] | None = None,
    index_name: str = "reference_idx",
    columns_name: str = "candidate_idx",
) -> pd.DataFrame:
    """
    Compute pairwise metric between all object pairs.

    Computes the given PairwiseHypothesisMetric for all (i, j) object pairs where
    objs[i] is the reference and objs[j] is the candidate, including i == j.

    The result is returned as a DataFrame with shape (len(objs), len(objs)), indexed
    by reference object index and column-labeled by candidate object index.

    Args:
        objs: A list of hypothesis-like objects.
        metric: A single pairwise metric to apply between all pairs.
        index_labels: Optional labels for the rows and columns (default: 0..len(objs)-1).
        index_name: Name for the row index (defaults to 'reference_idx').
        columns_name: Name for the column index (defaults to 'candidate_idx').

    Returns:
        pd.DataFrame: A square DataFrame of shape (N, N) where N = len(objs),
                      containing metric scores for all (i, j) pairs.

    """
    n = len(objs)
    labels = index_labels if index_labels is not None else list(range(n))

    if index_labels is not None and len(index_labels) != n:
        raise ValueError(f"index_labels must have length {n}, got {len(index_labels)}")

    data = []
    for _, ref in enumerate(objs):
        row = []
        for _, cand in enumerate(objs):
            score = metric.compute(ref, cand)
            row.append(score)
        data.append(row)

    df = pd.DataFrame(data, index=pd.Index(labels, name=index_name), columns=pd.Index(labels, name=columns_name))
    return df


def compute_pairwise_metric_against_reference(
    reference_obj: Any,
    target_objs: list[Any],
    metric: PairwiseHypothesisMetric[Any, Any],
    target_labels: list[Any] | None = None,
    target_label_name: str = "target",
) -> pd.DataFrame:
    """
    Compute a pairwise metric between a single reference object and multiple target objects.

    This is useful for comparing N candidate models to a single ground truth (or baseline),
    or comparing many samples to a reference sample.

    Args:
        reference_obj: The single object treated as reference in all comparisons.
        target_objs: A list of objects to compare to the reference.
        metric: A PairwiseHypothesisMetric instance implementing compute(reference, target).
        target_labels: Optional labels for the target objects (e.g., model names). Must match len(target_objs).
        target_label_name: Name to assign to the DataFrame index (e.g., "method", "sample").

    Returns:
        pd.DataFrame with one row per target and columns:
            - reference_obj
            - target_obj
            - metric value

    """
    if target_labels is not None and len(target_labels) != len(target_objs):
        raise ValueError(
            f"Length of target_labels ({len(target_labels)}) must match number of target_objs ({len(target_objs)})."
        )

    metric_name = type(metric).__name__
    records = []

    for i, target in enumerate(target_objs):
        row = {
            "reference_obj": reference_obj,
            "target_obj": target,
            metric_name: metric.compute(reference_obj, target),
        }
        records.append(row)

    index = pd.Index(target_labels if target_labels else range(len(target_objs)), name=target_label_name)
    df = pd.DataFrame(records, index=index)
    return df
