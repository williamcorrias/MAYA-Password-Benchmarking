import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from models.VGPT2.src.utils.param_scheduler import ExponentialScheduler

class Model(nn.Module):
    def __init__(self, parameter_schedulers, device):
        super(Model, self).__init__()

        self.device = device
        self.parameter_schedulers = parameter_schedulers
        self.learning_rate = parameter_schedulers["learning_rate"]

        self.global_step = 0

        max_steps = parameter_schedulers["max_steps"]
        decay_rate = parameter_schedulers["decay_rate"]
        max_value = parameter_schedulers["max_value"]

        self.parameter_schedulers["scheduler"] = ExponentialScheduler(max_steps, decay_rate, max_value)

    def clip_gradients(self, clip_value):
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value)

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        scheduler = self.init_scheduler(self.parameter_schedulers, optimizer)
        return optimizer, scheduler

    def init_scheduler(self, parameter_schedulers, optimizer):
        gamma = parameter_schedulers['gamma']
        step_size = parameter_schedulers['step_size']

        scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        return scheduler
