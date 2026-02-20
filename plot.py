import pathlib
import argparse
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from src.ablation import get_runs

def _get_param_from_run_name(run_name: str, param: str) -> str | None:
    """Extracts a parameter value from a run name string."""
    match = re.search(f"{param}=([^__]*)", run_name)
    if match:
        value = match.group(1)
        if value == "None":
            return None
        return value
    return None

def plot_results(
    log_dir: pathlib.Path,
    method: str | None = None,
    noise_std: str | None = None,
    group_by: str = "noiseSTD",
    performance_metric: str = "loss_01/test",
    title: str | None = None,
    out_path: pathlib.Path | None = None,
    show: bool = True,
):
    """
    Generates and displays a plot of test performance vs. epoch for a given method,
    averaged over different random seeds.

    Args:
        log_dir: The root directory containing the log files.
        method: The method to plot (e.g., 'isotropic', 'bineta'). If None, compare all methods.
        noise_std: If provided, only plot runs with this specific noise standard deviation.
        group_by: The parameter to group runs by (e.g., 'noiseSTD', 'numSAMPLES').
        performance_metric: The name of the scalar metric to plot from TensorBoard logs.
        title: The title of the plot.
        out_path: Path to save the generated plot image.
        show: Whether to display the plot in a window.
    """

    # 1. Prepare filters for get_runs
    filters = []
    if method is not None:
        filters.append(f"covarianceMODE={method}")
    if noise_std is not None:
        filters.append(f"noiseSTD={noise_std}")

    # 2. Get all runs matching the filters
    all_runs = get_runs(log_dir, *filters)


    if not all_runs:
        filter_str = " AND ".join(filters) if filters else "none"
        print(f"No runs found matching filters: '{filter_str}' in directory '{log_dir}'.")
        return

    # 2. Group runs by the specified parameter
    runs_by_group = defaultdict(list)
    for run_name, run_data in all_runs.items():
        if performance_metric not in run_data:
            print(f"Metric '{performance_metric}' not found in run '{run_name}'. Skipping.")
            continue
        
        param_value_str = _get_param_from_run_name(run_name, group_by)
        if param_value_str is None:
            print(f"Could not find parameter '{group_by}' in run '{run_name}'. Skipping.")
            continue
        
        try:
            # Attempt to convert to float, but keep as string if it fails (e.g., for 'None')
            param_value = float(param_value_str)
        except (ValueError, TypeError):
            param_value = param_value_str

        runs_by_group[param_value].append(run_data[performance_metric])

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_group_keys = sorted(runs_by_group.keys(), key=lambda x: (isinstance(x, str), x))

    for group_val in sorted_group_keys:
        runs = runs_by_group[group_val]
        
        # Ensure all runs for a given group have the same length by truncating to the minimum length
        min_len = min(len(run) for run in runs)
        runs_array = np.array([run[:min_len] for run in runs])

        mean = np.mean(runs_array, axis=0)
        std = np.std(runs_array, axis=0)
        epochs = np.arange(1, len(mean) + 1)

        line, = ax.plot(epochs, mean, label=f"{group_by}={group_val}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Test Performance ({performance_metric})")
    if title:
        ax.set_title(title)
    elif method is not None:
        ax.set_title(f"Performance of method '{method}' grouped by {group_by}")
    else:
        ax.set_title(f"Comparison of all methods grouped by {group_by}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # 4. Save and/or show the plot
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        print(f"Plot saved to {out_path}")

    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot average test performance for a given method across different parameter values."
    )
    parser.add_argument(
        "--logdir",
        default="logs_ablation",
        type=pathlib.Path,
        help="Root directory where run logs are stored.",
    )
    parser.add_argument(
        "--method",
        help="Optional: Method to plot (e.g., 'isotropic', 'bineta', 'inv_sq_grads').",
    )
    parser.add_argument(
        "--noise-std",
        help="Optional: Filter by a specific noise standard deviation (e.g., '0.01').",
    )
    parser.add_argument(
        "--group-by",
        default="noiseSTD",
        help="Parameter to group the runs by (e.g., 'noiseSTD', 'numSAMPLES').",
    )
    parser.add_argument(
        "--metric",
        default="loss_01/test",
        help="The performance metric to plot from TensorBoard scalars.",
    )
    parser.add_argument("--title", help="Optional title for the plot.")
    parser.add_argument("--out", type=pathlib.Path, help="Optional path to save the plot image.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    
    args = parser.parse_args()

    plot_results(
        log_dir=args.logdir,
        method=args.method,
        noise_std=args.noise_std,
        group_by=args.group_by,
        performance_metric=args.metric,
        title=args.title,
        out_path=args.out,
        show=not args.no_show,
    )

if __name__ == "__main__":
    main()
