from __future__ import annotations
import abc
import dataclasses
import json
import math
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def print_results_table(
    log_directory: str | pathlib.Path,
    *,
    x_axis: PlotAxis,
    y_axis: PlotAxis,
) -> None:
    log_directory_path = pathlib.Path(log_directory).resolve()
    runs = get_runs(log_directory_path)
    if not runs:
        print(f"No runs found under: {log_directory_path}")
        return

    config_by_equivalence_key, tests_by_equivalence_key = collect_equivalent_runs(log_directory_path, runs)
    if not config_by_equivalence_key:
        print("No runs with readable configuration and required scalars were found.")
        return

    all_configs = list(config_by_equivalence_key.values())
    cell_tests, x_axis_indices, y_axis_indices = pivot_cells(
        config_by_equivalence_key,
        tests_by_equivalence_key,
        all_configs,
        x_axis,
        y_axis,
    )
    if not cell_tests:
        print("No cells could be constructed (axes filtered everything).")
        return

    x_indices_ordered = sorted(x_axis_indices, key=x_axis.key_order)
    y_indices_ordered = sorted(y_axis_indices, key=y_axis.key_order)

    print_table(x_indices_ordered, y_indices_ordered, cell_tests, x_axis, y_axis)
    print_axis_sections(all_configs, x_axis, y_axis)
    print_best_x_index_by_y_index(x_indices_ordered, y_indices_ordered, cell_tests, x_axis, y_axis)


def collect_equivalent_runs(
    log_directory_path: pathlib.Path,
    runs: Runs,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[float]]]:
    config_by_equivalence_key: Dict[str, Dict[str, Any]] = {}
    tests_by_equivalence_key: Dict[str, List[float]] = defaultdict(list)

    for run_name, scalars in runs.items():
        config = read_config_from_run_directory(log_directory_path / run_name)
        if config is None:
            continue

        best_test = best_test_at_min_valid(scalars)
        if best_test is None:
            continue

        equivalence_key = config_equivalence_key(config)
        config_by_equivalence_key.setdefault(equivalence_key, config)
        tests_by_equivalence_key[equivalence_key].append(best_test)

    return config_by_equivalence_key, tests_by_equivalence_key


def pivot_cells(
    config_by_equivalence_key: Dict[str, Dict[str, Any]],
    tests_by_equivalence_key: Dict[str, List[float]],
    all_configs: Sequence[Dict[str, Any]],
    x_axis: PlotAxis,
    y_axis: PlotAxis,
) -> Tuple[Dict[Tuple[str, str], List[float]], set[str], set[str]]:
    cell_tests: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    x_axis_indices: set[str] = set()
    y_axis_indices: set[str] = set()

    for equivalence_key, config in config_by_equivalence_key.items():
        tests = tests_by_equivalence_key.get(equivalence_key, [])
        if not tests:
            continue

        try:
            x_index = x_axis.key(config, all_configs)
            y_index = y_axis.key(config, all_configs)
        except KeyError:
            continue

        if not isinstance(x_index, str) or not isinstance(y_index, str):
            raise TypeError("PlotAxis.key(config, all_configs) must return a string index")

        cell_tests[(y_index, x_index)].extend(tests)
        x_axis_indices.add(x_index)
        y_axis_indices.add(y_index)

    return cell_tests, x_axis_indices, y_axis_indices


def print_table(
    x_indices_ordered: List[str],
    y_indices_ordered: List[str],
    cell_tests: Dict[Tuple[str, str], List[float]],
    x_axis: PlotAxis,
    y_axis: PlotAxis,
) -> None:
    header, rows = build_table(x_indices_ordered, y_indices_ordered, cell_tests, x_axis, y_axis)
    column_widths = column_widths_for_table(header, rows)

    print(f"\nAveraged test error per number of samples (mean ± 1 std over seeds)\n")
    print(format_table_row(header, column_widths))
    print(table_separator(column_widths))
    for row in rows:
        print(format_table_row(row, column_widths))


def print_axis_sections(
    all_configs: Sequence[Dict[str, Any]],
    x_axis: PlotAxis,
    y_axis: PlotAxis,
) -> None:
    for axis in (x_axis, y_axis):
        section_text = axis.section(all_configs)
        if section_text:
            print(section_text)


def print_best_x_index_by_y_index(
    x_indices_ordered: List[str],
    y_indices_ordered: List[str],
    cell_tests: Dict[Tuple[str, str], List[float]],
    x_axis: PlotAxis,
    y_axis: PlotAxis,
) -> None:
    print(f"\nBest {x_axis.name()} per {y_axis.name()} (by mean):")
    for y_index in y_indices_ordered:
        best = best_x_index_for_y_index(y_index, x_indices_ordered, cell_tests)
        if best is None:
            print(f"  {y_index}: no data")
            continue
        mean, standard_deviation, x_index = best
        print(f"  {y_index}: {x_index} -> {mean:.4f} ± {standard_deviation:.4f}")


def best_x_index_for_y_index(
    y_index: str,
    x_indices_ordered: List[str],
    cell_tests: Dict[Tuple[str, str], List[float]],
) -> Optional[Tuple[float, float, str]]:
    best: Optional[Tuple[float, float, str]] = None
    for x_index in x_indices_ordered:
        tests = cell_tests.get((y_index, x_index), [])
        if not tests:
            continue
        mean, standard_deviation = mean_and_standard_deviation(tests)
        if best is None or mean < best[0]:
            best = (mean, standard_deviation, x_index)
    return best


def build_table(
    x_indices_ordered: List[str],
    y_indices_ordered: List[str],
    cell_tests: Dict[Tuple[str, str], List[float]],
    x_axis: PlotAxis,
    y_axis: PlotAxis,
) -> Tuple[List[str], List[List[str]]]:
    header = [f"{y_axis.name()} \\ {x_axis.name()}"] + x_indices_ordered
    rows: List[List[str]] = []
    for y_index in y_indices_ordered:
        rows.append(build_row_for_y_index(y_index, x_indices_ordered, cell_tests))
    return header, rows


def build_row_for_y_index(
    y_index: str,
    x_indices_ordered: List[str],
    cell_tests: Dict[Tuple[str, str], List[float]],
) -> List[str]:
    row = [y_index]
    for x_index in x_indices_ordered:
        tests = cell_tests.get((y_index, x_index), [])
        row.append(format_cell(tests))
    return row


def format_cell(tests: List[float]) -> str:
    if not tests:
        return "-"
    mean, standard_deviation = mean_and_standard_deviation(tests)
    return f"{mean:.4f} ± {standard_deviation:.4f}"


def format_table_row(row: List[str], column_widths: List[int]) -> str:
    return " | ".join(str(cell).ljust(width) for cell, width in zip(row, column_widths))


def column_widths_for_table(header: List[str], rows: List[List[str]]) -> List[int]:
    return [max(len(str(cell)) for cell in column) for column in zip(header, *rows)]


def table_separator(column_widths: List[int]) -> str:
    return "-+-".join("-" * width for width in column_widths)


def config_equivalence_key(config: Dict[str, Any]) -> str:
    canonical = canonical_config(config)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def canonical_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {field: config.get(field) for field in CANONICAL_CONFIG_FIELDS}


def read_config_from_run_directory(run_directory: pathlib.Path) -> Optional[Dict[str, Any]]:
    file_path = run_directory / CONFIG_JSON_FILENAME
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text())
    except ValueError:
        return None
    config = payload.get("cfg")
    return config if isinstance(config, dict) else None


def best_test_at_min_valid(scalars: Run) -> Optional[float]:
    valid = scalars.get(VALID_TAG)
    test = scalars.get(TEST_TAG)
    if not valid or not test:
        return None
    count = min(len(valid), len(test))
    if count == 0:
        return None
    best_index = min(range(count), key=lambda i: valid[i])
    return test[best_index]


def mean_and_standard_deviation(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    n = len(xs)
    if n == 0:
        return math.inf, math.inf

    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0

    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


VALID_TAG = "loss_01/valid"
TEST_TAG = "loss_01/test"
CONFIG_JSON_FILENAME = "best_loss.json"
CANONICAL_CONFIG_FIELDS = (
    "dataset",
    "noise_std",
    "num_noise_samples_batch",
    "num_noise_samples_accumulation",
    "covariance_mode",
    "noise_scheduler",
    "noise_scheduler_start_step_ratio",
    "noise_scheduler_end_step_ratio",
    "n_epochs",
    "lr",
)


@dataclasses.dataclass(frozen=True)
class PlotAxis(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        """
        Display name of the axis.
        """

    @abc.abstractmethod
    def key(self, cfg: Dict[str, Any], all_cfgs: Sequence[Dict[str, Any]]) -> str:
        """
        Bucket of cfg. Two cfgs in the same bucket are identified together along this axis.
        """

    def key_order(self, idx: str):
        """
        Customizable display order for cfg buckets.
        """
        return idx

    def section(self, all_cfgs: Sequence[Dict[str, Any]]) -> Optional[str]:
        """
        Extra section to add in the printed output. Useful to show which bucket map to which numerical config values.
        For example, certain covariance modes need a larger noise scale than others to obtain a similar effect on the plot.
            We can override this method to print a section that show the mapping for each covariance mode.
        """
        return None


@dataclasses.dataclass(frozen=True)
class FieldAxis(PlotAxis):
    field: str
    axis_name: Optional[str] = None

    def name(self) -> str:
        return self.axis_name or self.field

    def key(self, cfg: Dict[str, Any], all_cfgs: Sequence[Dict[str, Any]]) -> str:
        value = cfg.get(self.field)
        if value is None:
            raise KeyError(f"missing cfg[{self.field!r}]")
        return str(value)


@dataclasses.dataclass(frozen=True)
class NoiseLevelAxis(PlotAxis):
    covariance_mode_field: str = "covariance_mode"
    sigma_field: str = "noise_std"
    axis_name: str = "noise_level"
    eps: float = 1e-6

    def name(self) -> str:
        return self.axis_name

    def key(self, cfg: Dict[str, Any], all_cfgs: Sequence[Dict[str, Any]]) -> str:
        sigma = cfg.get(self.sigma_field)
        if sigma is None:
            return "none"

        covariance_mode = cfg[self.covariance_mode_field]
        buckets = self._build_sigma_buckets(all_cfgs, covariance_mode)
        for i, bucket_value in enumerate(buckets):
            if abs(sigma - bucket_value) <= self.eps:
                return f"L{i}"

        raise KeyError(f"sigma {sigma} not within eps of any bucket rep for partition {covariance_mode!r}")

    def _build_sigma_buckets(self, all_cfgs, covariance_mode):
        sigmas = self._collect_sigmas(all_cfgs, covariance_mode)
        if not sigmas:
            raise KeyError(f"no sigmas found for partition {covariance_mode!r}")

        sigmas = sorted(sigmas)
        buckets = [sigmas[0]]
        last = sigmas[0]
        for x in sigmas[1:]:
            if abs(x - last) <= self.eps:
                continue
            buckets.append(x)
            last = x

        return buckets

    def _collect_sigmas(self, all_cfgs, covariance_mode):
        sigmas = []

        for c in all_cfgs:
            if c.get(self.covariance_mode_field) != covariance_mode:
                continue
            v = c.get(self.sigma_field)
            if v is None:
                continue
            sigmas.append(float(v))

        return sigmas

    def key_order(self, idx: str):
        if idx == "none":
            return (-1,)
        if idx.startswith("L"):
            try:
                return 0, int(idx[1:])
            except ValueError:
                pass
        return 1, idx

    def section(self, all_cfgs: Sequence[Dict[str, Any]]) -> Optional[str]:
        covariance_modes = sorted(
            {str(c.get(self.covariance_mode_field)) for c in all_cfgs if c.get(self.covariance_mode_field) is not None}
        )
        if not covariance_modes:
            return None

        lines = ["", "Sigma values per level (per method):"]

        any_levels = False
        for covariance_mode in covariance_modes:
            sigmas = self._collect_sigmas(all_cfgs, covariance_mode)
            if not sigmas:
                lines.append(f"  {covariance_mode}: (no sigma levels)")
                continue

            any_levels = True
            buckets = self._build_sigma_buckets(all_cfgs, covariance_mode)
            parts = [f"L{i}={sigma:g}" for i, sigma in enumerate(buckets)]
            lines.append(f"  {covariance_mode}: " + ", ".join(parts))

        if not any_levels:
            return None

        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class NumSamplesAxis(PlotAxis):
    batch_field: str = "num_noise_samples_batch"
    accumulation_field: str = "num_noise_samples_accumulation"
    sigma_field: str = "noise_std"

    def name(self) -> str:
        return "num_samples"

    def key(self, cfg: Dict[str, Any], all_cfgs: Sequence[Dict[str, Any]]) -> str:
        noise_std = cfg[self.sigma_field]
        if noise_std is None:
            return "none"

        batch = cfg[self.batch_field]
        accumulation = cfg[self.accumulation_field]
        return str(batch * accumulation)

    def key_order(self, idx: str):
        try:
            return int(idx)
        except ValueError:
            return -1


def get_runs(logdir: str | pathlib.Path, *run_name_filters: str) -> Dict[str, Run]:
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

        scalars = read_scalars_from_run(run_dir)
        if scalars:
            result[run_name] = scalars

    return result


def read_scalars_from_run(run_dir: pathlib.Path) -> Dict[str, List[float]]:
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


Run = Dict[str, List[float]]
Runs = Dict[str, Run]
