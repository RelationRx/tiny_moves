from typing import Any

import pandas as pd

from tiny_moves.metrics.single_hypothesis_metrics.template import (
    SingleHypothesisMetric,
)


def compute_single_hypothesis_metrics_on_objects(
    objs: list[Any],
    metrics: list[SingleHypothesisMetric[Any, Any]],
    index_labels: list[Any] | None = None,
    index_name: str = "trajectory_step",
) -> pd.DataFrame:
    """
    Compute single-hypothesis metrics on a provided trajectory of hypothesis-like objects.

    This function accepts a list of objects (for example, Hypothesis or GeneModuleTheme instances),
    computes the specified metrics for each object, and returns a DataFrame with one row per object.
    The DataFrame will include:
      • 'hypothesis_obj': the original object
      • one column per metric, named by the metric's class name

    Args:
        objs (List[Any]): A list of hypothesis-like objects in trajectory order.
        metrics (List[SingleHypothesisMetric[Any]]): List of metrics to compute for each object.
        index_labels (List[Any] | None): Optional list of labels for each trajectory step.
            If provided, must be same length as `objs`. If None, a default RangeIndex is used.
        index_name (str): Name to assign to the Index. Defaults to "trajectory_step".

    Returns:
        pd.DataFrame: A DataFrame with index labeled by `index_labels` (or RangeIndex),
                      one column 'hypothesis_obj', plus one column per metric.

    """
    # 1) Build one record per object
    records = []
    for obj in objs:
        row: dict[str, Any] = {"hypothesis_obj": obj}
        for metric in metrics:
            metric_name = type(metric).__name__
            row[metric_name] = metric.compute(obj)
        records.append(row)

    # 2) Determine index
    if index_labels is not None:
        if len(index_labels) != len(records):
            raise ValueError(
                f"Length of index_labels ({len(index_labels)}) must match number of objects ({len(records)})."
            )
        idx = pd.Index(index_labels, name=index_name)
    else:
        idx = pd.Index(range(len(records)), name=index_name)

    # 3) Construct DataFrame
    df = pd.DataFrame.from_records(records, index=idx)
    return df
