import logging
from abc import ABC, abstractmethod

logger = logging.getLogger()

import numpy as np


class Scheduler(ABC):
    @abstractmethod
    def __call__(self, step) -> float:
        ...


class ExponentialScheduler(Scheduler):
    """
    Value follows the cumulative normal distribution.
    Scheduled value increases from 0.0 to max_value
    max_steps: step for which half of max_value is reached
    decay_rate: lower for flatter plot, higher for sharper plot (more like step function)
    """

    def __init__(self, max_steps: int = 1000, decay_rate: float = 0.1, max_value: float = 1.0):
        self.max_steps = max_steps
        self.decay_rate = decay_rate
        self.max_value = max_value

    def __call__(self, step):
        return float(self.max_value / (1.0 + np.exp(-self.decay_rate * (step - self.max_steps))))


class LinearScheduler(Scheduler):
    """
    Linear progression from start_value (default 0.) to end_value (default 1.) in max_steps steps
    """

    def __init__(self, max_steps: int = 1000, start_value: float = 0.0, end_value: float = 1.0):
        self.max_steps = max_steps
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, step):
        value = self.start_value + (float(step) / self.max_steps) * (self.end_value - self.start_value)
        return min(1.0, value)


class ConstantScheduler(object):
    def __init__(self, value: float = 1.0):
        self.value = value

    def __call__(self, step):
        return self.value
