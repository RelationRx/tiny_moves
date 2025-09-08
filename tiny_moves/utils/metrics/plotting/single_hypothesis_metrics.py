import matplotlib.pyplot as plt
import pandas as pd


def plot_single_hypothesis_metric_evolution_across_trajectory(df: pd.DataFrame) -> None:
    """
    Plot the evolution of single hypothesis metrics across a trajectory.

    This function takes a DataFrame where:
      - the index is 'trajectory_step' (integer step)
      - one column is 'hypothesis_obj' (the Pydantic object, not plotted)
      - all other columns are metric names with numeric values

    It generates a line plot for each metric showing its value across the trajectory.
    Each plot will have the trajectory step on the x-axis (as integer ticks) and the metric value on the y-axis,
    with the y-axis forced to start at 0. X-tick labels are rotated 45° and right-aligned.

    """
    # Identify metric columns (exclude 'hypothesis_obj')
    metric_columns = [col for col in df.columns if col != "hypothesis_obj"]

    # Use the DataFrame index as the x-axis (trajectory steps)
    steps = list(df.index)

    # Plot each metric
    for metric in metric_columns:
        fig, ax = plt.subplots(figsize=(10, 5))  # you can adjust figsize as needed
        ax.plot(steps, df[metric], marker="o")
        ax.set_title(metric)
        ax.set_xlabel("Trajectory Step")
        ax.set_ylabel(metric)

        # Force integer ticks at exactly the steps
        ax.set_xticks(steps)

        # Rotate and right-align the x-tick labels
        ax.set_xticklabels(steps, rotation=45, ha="right")

        # Force y-axis to start at 0
        ax.set_ylim(bottom=0)

        ax.grid(True)

        # Make sure labels fit into the figure area
        plt.tight_layout()

    plt.show()


def plot_single_hypothesis_metrics_bar_comparison(df: pd.DataFrame, title: str | None = None) -> None:
    """
    Compare each metric across the rows of a DataFrame using bar plots.

    Args:
        df: DataFrame where the index is hypothesis labels and
                           all numeric columns are treated as metrics.
        title: Optional title for the plot. If None, no title is set.

    """
    # Only keep numeric columns (e.g., float/int)
    metric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    x_labels = [str(label) for label in df.index]

    for metric in metric_columns:
        values = df[metric].tolist()

        plt.figure()
        plt.bar(x_labels, values)
        plt.title(title) if title else plt.title(metric)
        plt.xlabel("Hypothesis")
        plt.ylabel(metric)
        plt.ylim(bottom=0)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

    plt.show()
