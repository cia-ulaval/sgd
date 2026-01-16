import json
import math
import pathlib
from collections import defaultdict


def _parse_noise_std(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_sigma(s):
    if s is None:
        return "none"

    if s == 0:
        return "0"

    exp = int(math.floor(math.log10(abs(s))))
    if -3 <= exp <= 3:
        return f"{s:g}"
    return f"{s:.2e}"


def print_results_table(log_dir: pathlib.Path):
    log_dir = pathlib.Path(log_dir)
    files = sorted(log_dir.rglob("best_loss.json"))
    if not files:
        print(f"No best_loss.json files found under: {log_dir}")
        return

    groups = defaultdict(list)

    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue

        cfg = d.get("cfg", {})
        method = cfg.get("covariance_mode", "unknown")
        sigma = _parse_noise_std(cfg.get("noise_std", None))

        test_01 = d.get("test_01", None)
        if test_01 is None:
            continue
        try:
            test_01 = float(test_01)
        except (TypeError, ValueError):
            continue

        groups[(method, sigma)].append(test_01)

    methods = sorted({m for (m, _s) in groups.keys()})
    sigmas = sorted({s for (_m, s) in groups.keys()},
                   key=lambda x: (x is None, float("inf") if x is None else x))

    table = []
    header = ["method \\ sigma"] + [_fmt_sigma(s) for s in sigmas]

    for m in methods:
        row = [m]
        for s in sigmas:
            vals = groups.get((m, s), [])
            if not vals:
                row.append("—")
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
            print(f"  {m}: {_fmt_sigma(s)} -> {mean:.4f} (n={n})")
