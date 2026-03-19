import json
import pathlib
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclasses.dataclass(frozen=True)
class Axis(abc.ABC):
    @abstractmethod
    def key(self, cfg: Dict[str, Any], all_cfgs: Sequence[Dict[str, Any]]) -> str:
        ...

    def key_order(self, idx: str) -> object:
        return idx


@dataclass(frozen=True)
class NumSamplesAxis(Axis):

    def key(self, cfg: Dict[str, Any], all_cfgs: Sequence[Dict[str, Any]]) -> str:
        b = cfg["num_noise_samples_batch"]
        a = cfg["num_noise_samples_accumulation"]
        return str(b * a)

    def sort_key(self, idx: str) -> object:
        # natural numeric order, but still string indices
        try:
            return int(idx)
        except ValueError:
            return idx


def get_runs(logdir: str | pathlib.Path, *run_name_filters: str) -> Dict[str, Dict[str, List[float]]]:
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


def _read_scalars_from_run(run_dir: pathlib.Path) -> Dict[str, List[float]]:
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
