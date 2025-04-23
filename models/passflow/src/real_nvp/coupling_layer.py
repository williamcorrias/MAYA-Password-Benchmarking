import torch
import torch.nn as nn

from enum import IntEnum
import numpy as np

from models.passflow.src.nn import MLP, ResNet

class MaskType(IntEnum):
    CHECKERBOARD = 0
    HORIZONTAL = 1
    CHAR_RUN = 2


class AffineTransform(nn.Module):
    def __init__(self, dim, device, mask_type, mask_pattern, net_type, n_hidden=2, hidden_size=256):
        """
        :param dim: dimension of x
        :param device: gpu or cpu
        :param mask_type: normal(left) or inverted (right) mask
        :param mask_pattern: tuple of (pattern type, param). Defines the pattern to use for splitting: horizontal splitting
        (split), checkerboard and checkerboard with runs of n chars (char_run). param defines the parameters of the split
        :param net_type: mlp or resnet
        :param n_hidden: number of hidden layers in s and t
        :param hidden_size: size of hidden layers in s and t
        """
        super().__init__()
        assert mask_type in {'left', 'right'}
        assert net_type in {'mlp', 'resnet'}

        self.dim = dim
        self.device = device
        self.mask = self.build_mask(mask_type=mask_type, mask_pattern=mask_pattern)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        if net_type == 'mlp':
            self.net = MLP(input_size=self.dim,
                           n_hidden=n_hidden,
                           hidden_size=hidden_size,
                           output_size=2 * self.dim)
        elif net_type == 'resnet':
            self.net = ResNet(in_features=self.dim,
                              hidden_features=hidden_size,
                              out_features=2 * self.dim,
                              num_blocks=n_hidden)
        else:
            raise NotImplementedError

    def build_mask(self, mask_type, mask_pattern):
        assert mask_type in {'left', 'right'}
        if mask_type == 'left':
            mask = self.get_mask_pattern(mask_pattern)
        elif mask_type == 'right':
            mask = 1 - self.get_mask_pattern(mask_pattern)
        else:
            raise NotImplementedError
        return mask

    def get_mask_pattern(self, mask_pattern, param=2):
        half_dim = int(self.dim / 2)
        if mask_pattern == MaskType.HORIZONTAL:
            return torch.FloatTensor(np.concatenate([np.ones(half_dim), np.zeros(half_dim)], axis=0)).to(self.device)
        elif mask_pattern == MaskType.CHECKERBOARD:
            return torch.FloatTensor(np.tile([0, 1], half_dim)).to(self.device)
        elif mask_pattern == MaskType.CHAR_RUN:
            if (self.dim / param) % 2:
                raise Exception(f'Cannot use char run mask of run {param} with feature vector of length {self.dim}.'
                                f'len(feature vector) / char_run must be even.')
            # TODO find a cleaner way to use param
            return torch.FloatTensor(np.tile([1] * param + [0] * param, int(self.dim / (2 * param)))).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, x, reverse=False):
        # returns transform(x), log_det
        batch_size = x.shape[0]
        mask = self.mask.repeat(batch_size, 1)
        x_ = x * mask

        log_s, t = self.net(x_).chunk(2, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)

        if reverse:  # inverting the transformation
            x = (x - t) * torch.exp(-log_s)
        else:
            x = x * torch.exp(log_s) + t
        return x, log_s
