from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pairwise_metric_evolution_across_trajectory(
    df: pd.DataFrame,
    separate_plots: bool = True,
    x_labels: list[Any] | None = None,
) -> None:
    """
    Plot the evolution of pairwise-hypothesis metrics across a trajectory of transitions.

    Args:
        df:
            - The index is something like "transition" (e.g. "step0→1", "step1→2", …).
            - Two columns are 'reference_obj' and 'candidate_obj' (Hypothesis objects, not plotted).
            - All other columns are metric names (numeric float scores).
        separate_plots:
            If True: draw one figure per metric.
            If False: overlay all metric-curves on a single figure with a legend.
        x_labels:
            Optional alternative labels for the x-axis. Must be the same length
            as the number of rows in `df`. If None, `df.index` is used.

    Behavior:
        • Automatically skips any non-numeric columns (e.g. 'reference_obj', 'candidate_obj').
        • Uses `x_labels` (if provided) or else `df.index` as the x-axis.
        • Forces the y-axis to start at 0.
        • Rotates x-tick labels by 45° (right-aligned).

    """
    # 1) Identify numeric metric columns (skip anything non-numeric)
    metric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if not metric_columns:
        raise ValueError("No numeric metric columns found to plot.")

    # 2) Determine what to use as x-axis labels
    num_rows = df.shape[0]
    if x_labels is None:
        x_vals = list(df.index)
    else:
        if len(x_labels) != num_rows:
            raise ValueError(f"Length of x_labels ({len(x_labels)}) must match number of rows in df ({num_rows}).")
        x_vals = x_labels

    # 3) Plotting
    if separate_plots:
        # One figure per metric
        for metric in metric_columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_vals, df[metric], marker="o", label=metric)
            ax.set_title(f"{metric} Across Trajectory")
            ax.set_xlabel("Transition")
            ax.set_ylabel(metric)

            ax.set_xticks(x_vals)
            ax.set_xticklabels(x_vals, rotation=45, ha="right")

            ax.set_ylim(bottom=0)
            ax.grid(True)
            plt.tight_layout()

    else:
        # Single figure with all metrics overlaid
        fig, ax = plt.subplots(figsize=(10, 5))
        for metric in metric_columns:
            ax.plot(x_vals, df[metric], marker="o", label=metric)
        ax.set_title("All Metrics Across Trajectory")
        ax.set_xlabel("Transition")
        ax.set_ylabel("Metric Value")

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_vals, rotation=45, ha="right")

        ax.set_ylim(bottom=0)
        ax.legend(loc="best")
        ax.grid(True)
        plt.tight_layout()

    plt.show()


def plot_pairwise_metric_heatmap(
    df: pd.DataFrame,
    metric_name: str,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 8),
    annot: bool = True,
    fmt: str = ".2f",
) -> None:
    """
    Plot a heatmap for pairwise metric values.

    Args:
        df: A square DataFrame of pairwise metric scores. Rows are references, columns are candidates.
        metric_name: Name of the metric, used for the title and colorbar label.
        cmap: Color map to use for the heatmap.
        figsize: Size of the figure.
        annot: Whether to annotate the cells with the metric values.
        fmt: Format string for annotations (e.g. ".2f" for 2 decimals).

    """
    plt.figure(figsize=figsize)
    sns.set(style="white", font_scale=1.1)

    ax = sns.heatmap(
        df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": metric_name},
        linewidths=0.5,
        square=True,
        linecolor="lightgray",
    )

    ax.set_title(f"Pairwise Metric: {metric_name}", fontsize=14, pad=12)
    ax.set_xlabel(str(df.columns.name or "Candidate"), fontsize=12)
    ax.set_ylabel(str(df.index.name or "Reference"), fontsize=12)
    plt.tight_layout()
    plt.show()
