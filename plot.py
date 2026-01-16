"""
helper to render plots from tensorboard logs dir
"""
import os
from glob import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_scalar_series(event_file, tag):
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()

    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        raise KeyError(
            f"Tag '{tag}' not found in {event_file}. Available scalar tags: {scalars}"
        )

    events = ea.Scalars(tag)
    steps = [1 + e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def find_event_files(logdir, suffixes):
    patterns = [os.path.join(logdir, "**", f"events.*{s}") for s in suffixes]
    files = []
    for p in patterns:
        files.extend(glob(p, recursive=True))
    return sorted(set(files))


def main(logdir, suffixes, labels=None, out_path=None, show=True):
    event_files = find_event_files(logdir, suffixes)
    event_files.reverse()

    if len(event_files) < 2:
        raise RuntimeError(
            f"Expected at least 2 matching event files in {logdir}, found {len(event_files)}"
        )

    event_files = event_files[-2:]

    if labels is None:
        labels = [f"run {i}" for i in range(len(event_files))]

    colors = ["blue", "orange"]

    fig, axes = plt.subplots(1, 2, figsize=(2 * 8.6, 4.4), sharex=True)
    ax_train, ax_valid = axes

    for idx, event_file in enumerate(event_files):
        steps_valid, values_valid = load_scalar_series(event_file, "loss/valid")
        ax_valid.plot(steps_valid, values_valid, label=labels[idx], color=colors[idx])

        steps_train, values_train = load_scalar_series(event_file, "loss/train")
        ax_train.plot(steps_train, values_train, label=labels[idx], color=colors[idx])

    ax_valid.set_xlabel("Époque")
    ax_valid.set_ylabel("Erreur de validation")
    ax_valid.grid(True)
    ax_valid.legend()

    ax_train.set_xlabel("Époque")
    ax_train.set_ylabel("Erreur d'entreinement")
    ax_train.grid(True)
    ax_train.legend()

    fig.suptitle("Comparaison entre la descente classique et la descente bruitée sur CIFAR100")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is not None:
        fig.savefig(out_path, dpi=300)

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training/validation loss from TensorBoard event files.")
    parser.add_argument("--logdir", default="logs_ablation", help="Root directory to search under.")
    parser.add_argument("--suffix", "-s", action="append", required=True, help="Suffix to match (can be passed multiple times). Example: -s 0 -s 0.01")
    parser.add_argument("--label", "-l", action="append", help="Label for each suffix (same count as --suffix).")
    parser.add_argument("--out", default=None, type=str, help="Output image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window; just save the figure.")
    args = parser.parse_args()

    if args.label is not None and len(args.label) != len(args.suffix):
        raise SystemExit("--label must be provided the same number of times as --suffix")

    main(
        logdir=args.logdir,
        suffixes=args.suffix,
        labels=args.label,
        out_path=args.out,
        show=not args.no_show,
    )
