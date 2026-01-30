import pathlib
import argparse
import matplotlib.pyplot as plt
import glob
from typing import List
from tensorboard.backend.event_processing import event_accumulator


def main():
    parser = argparse.ArgumentParser(description="Plot training/validation loss from TensorBoard event files.")
    parser.add_argument("--logdir", default="logs_ablation", type=pathlib.Path, help="Root directory to search under.")
    parser.add_argument("--suffix", "-s", action="append", required=True, help="Suffix to match (can be passed multiple times). Example: -s 0 -s 0.01")
    parser.add_argument("--label", "-l", action="append", required=True, help="Label for each suffix (same count as --suffix).")
    parser.add_argument("--title", "-t", default="Comparaison entre la descente classique et la descente bruitée sur CIFAR100", help="Label for each suffix (same count as --suffix).")
    parser.add_argument("--out", "-o", default=None, type=pathlib.Path, help="Output image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window; just save the figure.")
    args = parser.parse_args()

    if len(args.label) != len(args.suffix):
        raise SystemExit("Each --suffix <suffix> must be matched by a --label <label>.")

    generate_graphs(
        logdir=args.logdir,
        suffixes=args.suffix,
        labels=args.label,
        title=args.title,
        out_path=args.out,
        show=not args.no_show,
    )


def generate_graphs(logdir: pathlib.Path, suffixes: List[str], labels: List[str], title: str, out_path: pathlib.Path, show: bool):
    event_files = find_event_files(logdir, suffixes)
    colors = ["blue", "orange", "purple", "yellow"]

    fig, axes = plt.subplots(1, 2, figsize=(2 * 8.6, 4.4), sharex=True)
    ax_train, ax_valid = axes

    for idx, event_file in enumerate(event_files):
        steps_valid, values_valid = load_scalar_series(event_file, "loss/valid")
        ax_valid.plot(steps_valid, values_valid, label=labels[idx], color=colors[idx % len(colors)])

        steps_train, values_train = load_scalar_series(event_file, "loss/train")
        ax_train.plot(steps_train, values_train, label=labels[idx], color=colors[idx % len(colors)])

    ax_valid.set_xlabel("Époque")
    ax_valid.set_ylabel("Erreur de validation")
    ax_valid.grid(True)
    ax_valid.legend()

    ax_train.set_xlabel("Époque")
    ax_train.set_ylabel("Erreur d'entreinement")
    ax_train.grid(True)
    ax_train.legend()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is not None:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=300)

    if show:
        plt.show()


def find_event_files(logdir: pathlib.Path, suffixes: List[str]):
    event_files: List[pathlib.Path] = []
    for suffix in suffixes:
        found_files = list(logdir.rglob(f"*{glob.escape(suffix)}/events.*"))
        if not found_files:
            raise RuntimeError(
                f"Expected at least 1 matching event file in {logdir} for suffix '{suffix}'."
            )

        event_files.append(max(found_files, key=lambda p: p.stat().st_mtime))
    return event_files


def load_scalar_series(event_file: pathlib.Path, tag: str):
    event_accu = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    event_accu.Reload()

    scalars = event_accu.Tags().get("scalars", [])
    if tag not in scalars:
        raise KeyError(
            f"Tag '{tag}' not found in {event_file}. Available scalar tags: {scalars}"
        )

    events = event_accu.Scalars(tag)
    steps = [1 + e.step for e in events]
    values = [e.value for e in events]
    return steps, values


if __name__ == "__main__":
    main()
