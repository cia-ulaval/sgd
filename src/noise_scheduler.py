import abc
import math


class NoiseScheduler(abc.ABC):
    @abc.abstractmethod
    def step(self) -> None:
        ...

    @abc.abstractmethod
    def get_noise_scalar(self) -> float:
        ...

    def update_from_losses(self, train: float, valid: float) -> None:
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
    def __init__(self):
        self.current_scalar = 1.0

    def step(self):
        pass

    def get_noise_scalar(self) -> float:
        return self.current_scalar

    def update_from_losses(self, train: float, valid: float) -> None:
        self.current_scalar = math.exp(valid - train)
