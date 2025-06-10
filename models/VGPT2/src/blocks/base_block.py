import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, args, device, is_recurrent=False):
        super().__init__()

        assert not (args["embedding"] is None and args["embedding_dim"] is None)

        self.embedding = args["embedding"] or torch.nn.Embedding(args["vocab_dim"], args["embedding_dim"])
        self.embedding_dim = self.embedding.embedding_dim

        self.latent_dim = args["latent_dim"]
        self.vocab_dim = args["vocab_dim"]
        self.max_sequence_length = args["max_sequence_length"]
        self.sos_index = args["sos_index"]
        self.eos_index = args["eos_index"]
        self.pad_index = args["pad_index"]
        self.unk_index = args["unk_index"]

        self.device = device

        self.is_recurrent = is_recurrent

    def forward(self, x, lens):
        ...


class Decoder(torch.nn.Module):
    def __init__(self, args, device, embedding_dim=None, is_recurrent=False):
        super().__init__()
        self.latent_dim = args["latent_dim"]

        self.vocab_dim = args["vocab_dim"]
        self.max_sequence_length = args["max_sequence_length"]
        self.sos_index = args["sos_index"]
        self.eos_index = args["eos_index"]
        self.pad_index = args["pad_index"]
        self.unk_index = args["unk_index"]

        self.embedding = args["embedding"]
        self.embedding_dim = embedding_dim
        self.encoder = args["encoder"]

        self.device = device

        self.is_recurrent = is_recurrent

    def forward(self, z, x, lens):
        ...
