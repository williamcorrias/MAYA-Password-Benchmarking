import torch
import os

from models.VGPT2.src.models.loss import kl_divergence
from models.VGPT2.src.blocks.transformer_blocks import GPT2Encoder, GPT2Decoder
from models.VGPT2.src.blocks.base_block import Decoder, Encoder
from models.VGPT2.src.models.base_model import Model
from models.VGPT2.src.utils.helper import OneHotEncoding

class AE(Model):
    def __init__(self, args, train_embeddings, coupled_decoder, device):

        super().__init__(parameter_schedulers=args["parameter_schedulers"], device=device)

        self.vocab_dim = args["vocab_dim"]
        self.latent_dim = args["latent_dim"]
        self.embedding_dim = args["embedding_dim"]
        self.train_embeddings = train_embeddings
        self.max_sequence_length = args["max_sequence_length"]

        self.sos_index = args["sos_index"]
        self.eos_index = args["eos_index"]
        self.pad_index = args["pad_index"]
        self.unk_index = args["unk_index"]

        if not self.train_embeddings:
            self.embedding = OneHotEncoding(self.vocab_dim)
        else:
            self.embedding = torch.nn.Embedding(self.vocab_dim, self.embedding_dim, self.pad_index)
        self.embedding_dim = self.embedding.embedding_dim

        encoder_args = {"vocab_dim": self.vocab_dim, "max_sequence_length": self.max_sequence_length,
                        "latent_dim": self.latent_dim, "sos_index": self.sos_index, "eos_index": self.eos_index,
                        "pad_index": self.pad_index, "unk_index": self.unk_index, "embedding_dim": self.embedding_dim,
                        "embedding": self.embedding, "n_layer": args["encoder"]['n_layer'],
                        "n_head": args["encoder"]['n_head']}
        self.encoder = GPT2Encoder(encoder_args, self.device)

        decoder_args = {"vocab_dim": self.vocab_dim, "max_sequence_length": self.max_sequence_length,
                        "latent_dim": self.latent_dim, "sos_index": self.sos_index, "eos_index": self.eos_index,
                        "pad_index": self.pad_index, "unk_index": self.unk_index, "embedding": self.embedding,
                        "encoder": self.encoder if coupled_decoder else None,  "n_layer": args["decoder"]['n_layer'],
                        "n_head": args["decoder"]['n_head']}
        self.decoder = GPT2Decoder(decoder_args, self.device)

    def cross_entropy_loss(self, y_pred, y_target, lens):
        batch_size = lens.shape[0]
        max_len = torch.max(lens).item()
        loss = torch.nn.functional.cross_entropy(y_pred, y_target, ignore_index=self.pad_index, reduction="none")
        loss = loss.to(self.device)
        mask = (y_target.view(batch_size, -1) != self.pad_index).float().to(self.device)
        loss = loss.view(batch_size, -1) * (mask.float() / (lens.view(batch_size, 1).float()))
        loss = loss.sum() / batch_size
        return loss

    def on_train_batch_start(self, batch):
        self.initialize_hidden_state(batch[0].shape[0])

    def on_train_batch_end(self):
        self.detach_history()

    def initialize_hidden_state(self, batch_size, enc=True, dec=True):
        if enc and self.encoder.is_recurrent:
            self.encoder.initialize_hidden_state(batch_size, self.device)
        if dec and self.decoder.is_recurrent:
            self.decoder.initialize_hidden_state(batch_size, self.device)

    def detach_history(self, enc=True, dec=True):
        if self.encoder.is_recurrent and enc:
            self.encoder.reset_history()
        if self.decoder.is_recurrent and dec:
            self.decoder.reset_history()

class VAE(AE):
    def __init__(self, args, device, train_embeddings=True, coupled_decoder=False, temperature=1.0, drop_char_rate=0.0, beta=1.0, min_logvar=-20.0 ):

        super().__init__(args, train_embeddings, coupled_decoder, device)

        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_sigma = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_sigma)

        self.temperature = temperature

        self.drop_char_rate = drop_char_rate

        self.latent_to_mean = torch.nn.Linear(self.encoder.latent_dim, self.latent_dim)
        self.latent_to_logvar = torch.nn.Linear(self.encoder.latent_dim, self.latent_dim)

        self.beta = beta

        self.min_logvar = min_logvar

    def latent_to_sigma(self, z):
        logvar = self.latent_to_logvar(z)

        if self.min_logvar is not None:
            logvar = logvar.clamp(min=self.min_logvar)

        return torch.exp(0.5 * logvar)

    @property
    def hidden_dim(self):
        return self.decoder.embedding_dim

    def forward(self, x, lens):
        """
        returns: loss and KL divergences of VAE
        input & target shape: [B, T]
        Notation. B: batch size; T: seq len (== fix_len); V: voc size
        """
        if self.drop_char_rate > 0:
            x = self.drop_chars(x, lens)

        z = self.encoder(x, lens)

        mean, sigma = self.latent_to_mean(z), self.latent_to_sigma(z)

        z = torch.randn_like(mean, requires_grad=False)
        z = (mean + z * sigma).to(self.device)

        logits = self.decoder(z=z, x=x, lens=lens)  # [B, T, V]
        return logits, mean, sigma

    def drop_chars(self, x, lens):
        prob = torch.rand(*x.shape, device=self.device)
        prob[(x == self.sos_index) | (x == self.pad_index)] = 1.0

        x = x.clone()
        x[prob < (self.drop_char_rate / lens.unsqueeze(-1).float())] = self.unk_index
        return x

    def loss(self, y_pred, y_target, lens, mean, sigma, beta):
        y_pred = y_pred.to(self.device)
        y_target = y_target.to(self.device)
        lens = lens.to(self.device)
        mean = mean.to(self.device)
        sigma = sigma.to(self.device)

        if "scheduler" in self.parameter_schedulers:
            beta = self.parameter_schedulers["scheduler"](self.global_step)
        else:
            beta = self.beta

        reconstruction_loss = self.cross_entropy_loss(y_pred, y_target, lens)
        distance_loss = kl_divergence(mean, sigma)
        loss = reconstruction_loss + beta * distance_loss

        return loss, reconstruction_loss, distance_loss

    def encode(self, input_seq, seq_len):
        with torch.no_grad():
            B = input_seq.size(0)
            self.initialize_hidden_state(B, self.device)
            z, m, s = self.encoder((input_seq, seq_len))
        return z, m, s

    def process_batch(self, batch):
        x, lens = batch

        x.to(self.device)
        lens.to(self.device)

        batch_size, sequence_length = x.shape

        lens, sort_indices = torch.sort(lens, descending=True)
        x = x[sort_indices]

        y = x[:, 1:]
        x = x[:, :-1]

        logits, mean, sigma = self.forward(x, lens)
        logits = logits.to(self.device)
        mean = mean.to(self.device)
        sigma = sigma.to(self.device)

        logits = logits.contiguous().view(-1, self.vocab_dim)  # [T * B, V]

        loss, reconstruction_loss, distance_loss = self.loss(
            y_pred=logits.view(-1, self.vocab_dim),
            y_target=y.reshape(-1),
            lens=lens.reshape(batch_size, 1),
            mean=mean,
            sigma=sigma,
            beta=None,
        )
        pred = logits.argmax(dim=1).view(batch_size, -1)
        return {"loss": loss, "pred": pred, "target": y, "samples": None}

    def training_step(self, batch):
        self.on_train_batch_start(batch)
        step_results = self.process_batch(batch)
        self.on_train_batch_end()
        return step_results

    def sample_sequences(self, batch_size):
        z: torch.Tensor = self.sample_latent(batch_size)
        generated_sequences, _ = self.decoder.generate(z)
        return generated_sequences

    def test_step(self, batch):
        output = self.process_batch(batch)
        return output

    def predict_step(self, batch_size):
        generated_sequences = self.sample_sequences(batch_size)
        return {"loss": None, "pred": None, "target": None, "samples": generated_sequences}

    def sample_latent(self, batch_size):
        return self.prior.sample((batch_size,)).to(self.device)

    def generate(self, n):
        z = self.sample_latent(n)
        return self.decoder.generate(z)
