import math
from typing import List, Optional

import torch


def kl_divergence_by_distribution(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    prior_mean: Optional[torch.Tensor] = None,
    prior_sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    prior_mean = prior_mean if prior_mean is not None else torch.zeros_like(mean).device(mean.device)
    prior_sigma = prior_sigma if prior_sigma is not None else torch.ones_like(sigma).device(sigma.device)
    dist = torch.distributions.Normal(mean, sigma + 0.001)
    prior_dist = torch.distributions.Normal(prior_mean, prior_sigma)
    return torch.distributions.kl_divergence(dist, prior_dist).mean()


def kl_divergence(mean: torch.Tensor, sigma: torch.Tensor, reduction="mean") -> torch.Tensor:
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """

    kl = -0.5 * (1 + 2.0 * torch.log(sigma) - mean * mean - sigma * sigma)  # [B, D]
    kl = torch.mean(kl, dim=1)
    if reduction == "mean":
        return torch.mean(kl)
    elif reduction == "sum":
        return torch.sum(kl)
    else:
        return kl
