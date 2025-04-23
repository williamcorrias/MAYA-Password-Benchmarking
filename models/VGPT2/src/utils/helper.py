import torch
from typing import Union

class OneHotEncoding(object):
    def __init__(self, encoding_size):
        self.encoding_size = encoding_size

    def __call__(self, indexes):
        one_hot = torch.nn.functional.one_hot(indexes, self.encoding_size)
        return one_hot

    @property
    def embedding_dim(self):
        return self.encoding_size

Embedding = Union[torch.nn.Embedding, OneHotEncoding]

def sample(dist, mode="sample", unk_index=None):
    """
    Auxiliary sampling method.
    """
    if mode in ["sample-no-unk", "greedy-no-unk"] and unk_index is None:
        raise ValueError("Unknown index for the <unk> token!")
    if mode == "greedy":
        _, _sample = torch.topk(dist, 1, dim=-1)
    elif mode == "sample":
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == "sample-no-unk":
        # reduce chances for <unk>
        dist[:, :, unk_index] = dist.min()
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == "greedy-no-unk":
        # prevent <unk>
        dist[:, :, unk_index] = dist.min()
        _, _sample = torch.topk(dist, 1, dim=-1)
    else:
        raise ValueError(f"Unknown sampling mode = {mode}")

    _sample = _sample.squeeze()
    return _sample
