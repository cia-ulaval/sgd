import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Sequence
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def print_results_table(log_dir: pathlib.Path):
    files = sorted(log_dir.rglob("best_loss.json"))
    if not files:
        print(f"No best_loss.json files found under: {log_dir}")
        return

    groups = defaultdict(list)

    for fp in files:
        d = json.loads(fp.read_text())
        cfg = d["cfg"]
        method = cfg["covariance_mode"]
        sigma = cfg["noise_std"]
        test_01 = d["test_01"]
        groups[(method, sigma)].append(test_01)

    methods = sorted({m for (m, _s) in groups.keys()})
    sigmas = sorted({s for (_m, s) in groups.keys()}, key=lambda x: (x is None, float("inf") if x is None else x))

    table = []
    header = ["method \\ sigma"] + [str(s) for s in sigmas]

    for m in methods:
        row = [m]
        for s in sigmas:
            vals = groups.get((m, s), [])
            if not vals:
                row.append("-")
                continue
            mean = sum(vals) / len(vals)
            row.append(f"{mean:.4f} (n={len(vals)})")
        table.append(row)

    col_widths = [max(len(str(x)) for x in col) for col in zip(header, *table)]

    def fmt_row(r):
        return " | ".join(str(x).ljust(w) for x, w in zip(r, col_widths))

    print("\nAveraged test 0-1 error at best-valid epoch (mean over seeds)\n")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for r in table:
        print(fmt_row(r))

    print("\nBest sigma per method (by mean test error):")
    for m in methods:
        best = None
        for s in sigmas:
            vals = groups.get((m, s), [])
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            if best is None or mean < best[0]:
                best = (mean, s, len(vals))
        if best is None:
            print(f"  {m}: no data")
        else:
            mean, s, n = best
            print(f"  {m}: {s} -> {mean:.4f} (n={n})")


def get_runs(
    logdir: str | pathlib.Path,
    *run_name_filters: str,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Filters runs by substring match on the run directory name.

    Example:
      runs = get_runs("logs", "covarianceMODE=isotropic", "noiseSTD=0.04")
      # runs: {"...noiseSTD=0.04_covarianceMODE=isotropic...": {"loss_01/train": [values...], ...}, ...}.

    Args:
      logdir: root directory containing run subdirectories.
      *run_name_filters: all must appear in the run directory name for it to match.

    Returns:
      Dict[run_name, Dict[scalar_tag, List[float]]]
    """
    if isinstance(logdir, str):
        logdir = pathlib.Path(logdir).resolve()

    if not logdir.exists():
        raise FileNotFoundError(f"logdir does not exist: {logdir}")

    result = {}
    for run_dir in sorted([p for p in logdir.iterdir() if p.is_dir()]):
        run_name = run_dir.name
        if run_name_filters and not all(s in run_name for s in run_name_filters):
            continue

        scalars = _read_scalars_from_run(run_dir)
        if scalars:
            result[run_name] = scalars

    return result


def _read_scalars_from_run(
    run_dir: pathlib.Path,
) -> Dict[str, List[float]]:
    """
    Returns: dict {scalar_tag: [v0, v1, ...]} for one run.
    - scalar_whitelist: if provided, only include scalar tags in this list.
    - size_guidance_scalars:
        EventAccumulator size guidance for scalars.
        0 means "load all scalar events" (often what you want).
        You can set e.g. 10_000 to cap memory if logs are huge.
    """
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        return {}

    acc = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    acc.Reload()

    scalar_tags = acc.Tags().get("scalars", [])
    result = {}
    for tag in scalar_tags:
        events = acc.Scalars(tag)
        result[tag] = [float(e.value) for e in events]
    return result
