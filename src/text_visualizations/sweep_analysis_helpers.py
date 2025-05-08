# load subfolders
from collections import defaultdict
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from text_visualizations.plotting import plot_tsne_colors


def load_results_in_subdirs(sweep_summary, variables_path, saving_path):
    """Loads results of different parameter combinations stored in subdirectories."""

    results = copy.deepcopy(sweep_summary["experiment_results"])

    for exp in results:
        dir_name = ""
        if exp["success"]:
            for key in exp["params"].keys():
                dir_name = dir_name + f"{key.split(".")[-1]}-{exp["params"][key]}_"
            dir_path = Path(dir_name[:-1])
            exp["df_log_training"] = pd.read_hdf(
                variables_path / saving_path / dir_path / "df_log_training.h5"
            )
            exp["embeddings_2d"] = np.load(
                variables_path / saving_path / dir_path / "embeddings_2d.npy"
            )

        else:
            print(f"Experiment {exp["params"]} failed.")

    return results


# construct df
def construct_results_df(results, i=-1):
    """Constructs comparison df from the hyperparameter sweep."""
    dict_sweep_results = defaultdict(list)
    for exp in results:
        if exp["success"]:
            for key in exp["params"].keys():
                dict_sweep_results[key].append(exp["params"][key])
            dict_sweep_results["loss"].append(exp["df_log_training"].loss.iloc[i])
            dict_sweep_results["knn"].append(exp["df_log_training"].knn.iloc[i])

    return pd.DataFrame(dict_sweep_results)


# Assuming your dataframe looks like:
# param1, param2, metric1, metric2, ...
# where each row has different combinations of param1 and param2


# construct df
def plot_2d_embeddings(
    results, colors_iclr, figures_path, save_figs=False, exp_name=None
):
    assert not (save_figs == True) & (
        exp_name is None
    ), "You need to pass the exp_name too"
    for exp in results:
        dir_name = ""
        if exp["success"]:
            for key in exp["params"].keys():
                dir_name = dir_name + f"{key.split(".")[-1]}-{exp["params"][key]}_"
            fig, ax = plot_tsne_colors(
                exp["embeddings_2d"], colors_iclr, figsize=(4, 4)
            )
            ax.set_title(dir_name[:-1])
            ax.text(
                0,
                0.01,
                f"2D knn acc:  {exp["df_log_training"].knn.iloc[-1]*100:.1f}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                size=7,
            )
            if save_figs:
                fig.savefig(
                    figures_path / f"embedding_{exp_name}_{dir_name[:-1]}_v1.png"
                )


def create_heatmap_matrices(df, param1_col, param2_col, metric_cols):
    """
    Transform a parameter sweep dataframe into matrices for heatmap visualization.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing parameter sweep results
    param1_col : str
        Name of the first parameter column
    param2_col : str
        Name of the second parameter column
    metric_cols : list of str
        Names of the metric columns to visualize

    Returns:
    --------
    dict
        Dictionary mapping metric names to their corresponding matrices
    """
    # Get unique values for each parameter
    param1_values = sorted(df[param1_col].unique())
    param2_values = sorted(df[param2_col].unique())

    # Create result matrices dictionary
    result_matrices = {}

    # For each metric, create a matrix
    for metric in metric_cols:
        # Initialize matrix with NaN values
        matrix = np.full((len(param1_values), len(param2_values)), np.nan)

        # Fill the matrix with values from the dataframe
        for i, p1 in enumerate(param1_values):
            for j, p2 in enumerate(param2_values):
                # Find the row with this parameter combination
                row = df[(df[param1_col] == p1) & (df[param2_col] == p2)]
                if not row.empty:
                    matrix[i, j] = row[metric].values[0]

        # Create a DataFrame for better labeling in the heatmap
        matrix_df = pd.DataFrame(matrix, index=param1_values, columns=param2_values)

        result_matrices[metric] = matrix_df

    return result_matrices


# Example usage:
# matrices = create_heatmap_matrices(df, 'learning_rate', 'batch_size', ['accuracy', 'loss'])


def plot_heatmaps(matrices, param1_name, param2_name, cmap="viridis"):
    """
    Plot heatmaps for each metric.

    Parameters:
    -----------
    matrices : dict
        Dictionary mapping metric names to their corresponding matrices
    param1_name : str
        Name of the first parameter for axis labeling
    param2_name : str
        Name of the second parameter for axis labeling
    cmap : str
        Colormap name
    """
    n_metrics = len(matrices)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    # If there's only one metric, axes won't be an array
    if n_metrics == 1:
        axes = [axes]

    for (metric_name, matrix), ax in zip(matrices.items(), axes):
        sns.heatmap(matrix, annot=True, cmap=cmap, ax=ax)
        ax.set_title(f"{metric_name}")
        ax.set_ylabel(param1_name)
        ax.set_xlabel(param2_name)
    return fig


# Example usage:
# plot_heatmaps(matrices, 'Learning Rate', 'Batch Size')
