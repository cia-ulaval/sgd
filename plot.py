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
    method: str,
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
        method: The method to plot (e.g., 'isotropic', 'bineta').
        performance_metric: The name of the scalar metric to plot from TensorBoard logs.
        title: The title of the plot.
        out_path: Path to save the generated plot image.
        show: Whether to display the plot in a window.
    """
    # 1. Get all runs for the specified method
    all_runs = get_runs(log_dir, f"covarianceMODE={method}")
    if not all_runs:
        print(f"No runs found for method '{method}' in log directory '{log_dir}'.")
        return

    # 2. Group runs by noise_std
    runs_by_noise = defaultdict(list)
    for run_name, run_data in all_runs.items():
        if performance_metric not in run_data:
            print(f"Metric '{performance_metric}' not found in run '{run_name}'. Skipping.")
            continue
        
        noise_std_str = _get_param_from_run_name(run_name, "noiseSTD")
        try:
            noise_std = float(noise_std_str) if noise_std_str is not None else "None"
            runs_by_noise[noise_std].append(run_data[performance_metric])
        except (ValueError, TypeError):
            print(f"Could not parse noiseSTD '{noise_std_str}' from run '{run_name}'. Skipping.")
            continue

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_noise_keys = sorted(runs_by_noise.keys(), key=lambda x: (isinstance(x, str), x))

    for noise_std in sorted_noise_keys:
        runs = runs_by_noise[noise_std]
        
        # Ensure all runs for a given noise_std have the same length by truncating to the minimum length
        min_len = min(len(run) for run in runs)
        runs_array = np.array([run[:min_len] for run in runs])

        mean = np.mean(runs_array, axis=0)
        std = np.std(runs_array, axis=0)
        epochs = np.arange(1, len(mean) + 1)

        line, = ax.plot(epochs, mean, label=f"noise_std={noise_std}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=line.get_color())

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Test Performance ({performance_metric})")
    ax.set_title(title if title else f"Performance of method '{method}'")
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
        description="Plot average test performance for a given method across different noise levels."
    )
    parser.add_argument(
        "--logdir",
        default="logs_ablation",
        type=pathlib.Path,
        help="Root directory where run logs are stored.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Method to plot (e.g., 'isotropic', 'bineta', 'inv_sq_grads').",
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
        performance_metric=args.metric,
        title=args.title,
        out_path=args.out,
        show=not args.no_show,
    )

if __name__ == "__main__":
    main()
