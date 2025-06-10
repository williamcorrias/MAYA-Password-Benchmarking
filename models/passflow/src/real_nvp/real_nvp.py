import torch
import torch.nn as nn

from enum import IntEnum


class SampleType(IntEnum):
    STATIC = 0
    DYNAMIC = 1


class RealNVP(nn.Module):
    def __init__(self,
                 dim,
                 device,
                 transforms):
        super().__init__()

        self.dim = dim
        self.device = device
        self.prior = torch.distributions.Normal(torch.tensor(0.).to(self.device), torch.tensor(1.).to(self.device))
        self.transforms = nn.ModuleList(transforms)
        self.sample_type = SampleType.STATIC


    def reset_prior(self):
        self.sample_type = SampleType.STATIC
        self.prior = torch.distributions.Normal(torch.tensor(0.).to(self.device), torch.tensor(1.).to(self.device))

    def set_prior(self, mean_vec, stddev_vec):
        self.sample_type = SampleType.DYNAMIC
        self.prior = torch.distributions.Normal(torch.FloatTensor(mean_vec).to(self.device),
                                                torch.FloatTensor(stddev_vec).to(self.device))

    def flow(self, x):
        # maps x -> z, and returns the log determinant (not reduced)
        z, log_det = x, torch.zeros_like(x)
        for op in self.transforms:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        return z, log_det

    def invert_flow(self, z):
        # z -> x (inverse of f)
        for op in reversed(self.transforms):
            z, _ = op.forward(z, reverse=True)
        return z

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return torch.sum(log_det, dim=1) + torch.sum(self.prior.log_prob(z), dim=1)

    def sample(self, num_samples=0):
        with torch.no_grad():
            if self.sample_type == SampleType.STATIC:
                z = self.prior.sample([num_samples, self.dim])
            else:
                z = self.prior.sample()
        return self.invert_flow(z)

    def nll(self, x):
        return - self.log_prob(x).mean()
