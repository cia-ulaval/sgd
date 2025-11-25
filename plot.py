import os
from glob import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

logdir = "logs_presentation"
tag = "loss/valid"


def load_scalar_series(event_file, tag):
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={"scalars": 0}  # load all scalar data
    )
    ea.Reload()

    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        raise KeyError(
            f"Tag '{tag}' not found in {event_file}. "
            f"Available scalar tags: {scalars}"
        )

    events = ea.Scalars(tag)  # list of Event(wall_time, step, value)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    event_files = glob(os.path.join(logdir, "**/events.*"))
    event_files.reverse()

    if len(event_files) < 2:
        raise RuntimeError(
            f"Expected at least 2 event files in {logdir}, found {len(event_files)}"
        )

    event_files = event_files[:2]

    labels = ["σ = 0", "σ = 0.01"]
    colors = ["blue", "orange"]

    plt.figure(figsize=(12.8, 4.8))

    for idx, event_file in enumerate(event_files):
        steps, values = load_scalar_series(event_file, tag)
        plt.plot(
            [s+1 for s in steps],
            values,
            label=labels[idx],
            color=colors[idx],
        )

    plt.xlabel("Époque")
    plt.ylabel("Erreur de validation")
    plt.title(f"Comparaison entre la descente classique et la descente bruitée sur CIFAR100")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = "loss_valid_comparison.png"  # or .pdf, .svg, etc
    plt.savefig(out_path, dpi=300)  # dpi controls resolution

    plt.show()


if __name__ == "__main__":
    main()
