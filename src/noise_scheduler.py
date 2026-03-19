import abc
import math
import optuna
import pathlib
from optuna.samplers import GPSampler
import pickle


class NoiseScheduler(abc.ABC):
    @abc.abstractmethod
    def step(self) -> None:
        ...

    @abc.abstractmethod
    def get_noise_scalar(self) -> float:
        ...

    def amend_config(self, cfg: dict) -> None:
        pass

    def update_from_losses(self, train: float, valid: float) -> None:
        pass

    def update_from_run(self, best_valid_loss: float) -> None:
        pass


class LinearNoiseScheduler(NoiseScheduler):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> None:
        self.current_step = min(self.current_step + 1, self.total_steps - 1)

    def get_noise_scalar(self):
        return (self.total_steps - 1 - self.current_step) / (self.total_steps - 1)


class PartialNoiseScheduler(NoiseScheduler):
    def __init__(self, total_steps: int, start_step_ratio: float, end_step_ratio: float):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_step_ratio = start_step_ratio
        self.end_step_ratio = end_step_ratio

    def step(self):
        self.current_step = min(self.current_step + 1, self.total_steps - 1)

    def get_noise_scalar(self) -> float:
        return float(self.start_step_ratio <= self.current_step / self.total_steps < self.end_step_ratio)


class TrainValidDiffNoiseScheduler(NoiseScheduler):
    def __init__(self, gamma: float):
        self.current_scalar = 1.0
        self.gamma = gamma

    def step(self):
        pass

    def get_noise_scalar(self) -> float:
        return self.current_scalar

    def update_from_losses(self, train: float, valid: float) -> None:
        self.current_scalar = math.exp((valid - train) * self.gamma)


class TrainValidDiffStaticPidNoiseScheduler(NoiseScheduler):
    def __init__(
        self,
        a_bounds: tuple[float, float],
        b_bounds: tuple[float, float],
        c_bounds: tuple[float, float],
        sigma_bounds: tuple[float, float],  
        n_startup_trials: int,
        seed: int,
        log_dir: pathlib.Path,
    ):
        seed_dir = log_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        self.sampler_path = seed_dir / "sampler.pkl"
        if self.sampler_path.exists():
            with open(self.sampler_path, "rb") as f:
                self.sampler = pickle.load(f)
        else:
            self.sampler = GPSampler(
                n_startup_trials=n_startup_trials,
                seed=seed,
            )
        self.study = optuna.create_study(
            direction="minimize",
            study_name="pid_params",
            load_if_exists=True,
            storage=f"sqlite:///{seed_dir / 'optuna_state.db'}",
            sampler=self.sampler,
        )

        self.a_bounds = a_bounds
        self.b_bounds = b_bounds
        self.c_bounds = c_bounds
        self.s_bounds = sigma_bounds

        self.current_scalar = 1.0
        self.trial = None
        self.a = None
        self.b = None
        self.c = None
        self.s = None

        self.reset_pid_state()
        self.ask_new_params()

    def reset_pid_state(self) -> None:
        self.current_scalar = 1.0
        self.prev_error = None
        self.integral = 0.0

    def ask_new_params(self) -> None:
        self.trial = self.study.ask()
        self.a = self.trial.suggest_float("a_raw", *self.a_bounds)
        self.b = self.trial.suggest_float("b_raw", *self.b_bounds)
        self.c = self.trial.suggest_float("c_raw", *self.c_bounds)
        self.s = self.trial.suggest_float("s_raw", *self.s_bounds)

    def step(self):
        pass

    def get_noise_scalar(self) -> float:
        return self.current_scalar

    def amend_config(self, config: dict) -> None:
        config["noise_std"] = self.get_base_sigma()
        config["noise_scheduler_trial"] = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "s": self.s,
        }

    def get_base_sigma(self) -> float:
        return self.softplus(self.s)

    def update_from_losses(self, train: float, valid: float) -> None:
        error = train - valid
        derivative = 0.0 if self.prev_error is None else error - self.prev_error
        self.integral += error

        pid = self.a * error + self.b * derivative + self.c * self.integral
        self.current_scalar = self.softplus(pid)
        self.prev_error = error

    def update_from_run(self, best_valid_loss: float) -> None:
        self.study.tell(self.trial, best_valid_loss)
        self.reset_pid_state()
        self.ask_new_params()

        with open(self.sampler_path, "wb") as f:
            pickle.dump(self.study.sampler, f)

    @staticmethod
    def softplus(x: float) -> float:
        if x > 20.0:
            return x
        return math.log1p(math.exp(x))
