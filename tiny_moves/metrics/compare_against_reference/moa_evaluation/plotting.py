from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tiny_moves.representations.structured_representations.triples import (
    ListOfEvaluationResults,
)


def plot_evaluation_heatmap(
    results: dict[str, ListOfEvaluationResults], attribute: str, title: str, savepath: None | Path = None
) -> None:
    """
    Generalized heatmap plotting function for visualizing evaluation attributes across methods.

    :param results: A dictionary mapping method names to their ListOfEvaluationResults
    :param attribute: The attribute of EvaluationResult to visualize (e.g., 'exists', 'correct_predicate',
    'correct_direction')
    :param title: Title of the heatmap
    """
    data = []
    for method, evaluation_list in results.items():
        for idx, result in enumerate(evaluation_list):
            triple = result.reference_triple
            triple_str = f"{triple.triple.subject} — {triple.triple.predicate} → {triple.triple.object}"
            score = getattr(result, attribute)
            data.append(
                {
                    "Triple": triple_str,
                    "Method": method,
                    "Score": int(score) if score is not None else 0.0,  # Fill None with 0
                }
            )

    df = pd.DataFrame(data)

    # Pivot the table for heatmap
    heatmap_data = df.pivot(index="Triple", columns="Method", values="Score").fillna(0)

    # Sort triples by total score
    triple_order = heatmap_data.sum(axis=1).sort_values(ascending=False).index
    # heatmap_data = heatmap_data.loc[triple_order]

    plt.figure(figsize=(12, max(6, 0.4 * len(triple_order))))
    _ = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap=sns.color_palette(["#f4f4f4", "#d3d3d3", "#1f77b4"]),
        linewidths=0.5,
        linecolor="gray",
        cbar=False,
        fmt=".1f",
    )
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Method")
    plt.ylabel("Reference Triple")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_evaluation_metrics(results: dict[str, Any], metrics: list[str] = [], palette: str = "Set2") -> None:
    """
    Plot per-metric bar charts for evaluation metrics across multiple methods.

    :param results: A dictionary where keys are method names and values are dicts of metric scores.
    :param palette: Matplotlib color palette for styling.
    """
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Method"})
    if not metrics:
        metrics = [col for col in df.columns if col != "Method"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        nrows=n_metrics, ncols=1, figsize=(10, 1.5 + 3 * n_metrics), sharex=True, constrained_layout=True
    )

    if n_metrics == 1:
        axes = [axes]

    colors = plt.get_cmap(palette)(range(len(df)))

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(df["Method"], df[metric], color=colors)
        ax.set_title(metric.replace("_", " ").capitalize(), fontsize=14, weight="semibold")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

        # Label bars with value
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.suptitle("Evaluation Metrics by Method", fontsize=16, weight="bold", y=1.02)
    plt.show()
