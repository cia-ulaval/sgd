import json
import pathlib
from collections import defaultdict


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
