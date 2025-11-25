import abc


class NoiseScheduler(abc.ABC):
    @abc.abstractmethod
    def step(self):
        ...

    @abc.abstractmethod
    def get_noise_scalar(self):
        ...


class LinearNoiseScheduler(NoiseScheduler):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step = min(self.current_step + 1, self.total_steps - 1)

    def get_noise_scalar(self):
        return (self.total_steps - 1 - self.current_step) / (self.total_steps - 1)
        # return 1 / math.sqrt(1 + self.alpha * self.current_step)
