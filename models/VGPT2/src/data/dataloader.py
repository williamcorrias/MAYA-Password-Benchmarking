import torch
import pickle
import re
import numpy as np

from models.VGPT2.src.tokenizers.char_tokenizer import CharTokenizer


class TokenizedTextDataLoader:
    def __init__(self, train_passwords, test_passwords, max_length, params):
        self.tokenizer = CharTokenizer(train_passwords, max_length)

        self.pad_index = self.tokenizer.pad_index
        self.sos_index = self.tokenizer.sos_index
        self.eos_index = self.tokenizer.eos_index
        self.unk_index = self.tokenizer.unk_index

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size = params["batch_size"]
        self.max_sequence_length = max_length

        self.train_passwords = [self.tokenizer.encode(data) for data in train_passwords]
        self.test_passwords = set(test_passwords)

    def get_batches(self, batch_size=128, is_train=True):
        data = self.train_passwords if is_train else self.test_passwords

        for i in range(0, len(data) - batch_size + 1, batch_size):
            batch = [torch.LongTensor(np.array(pwd)) for pwd in data[i:i + batch_size]]
            yield self.collate_text_batch(batch, self.tokenizer.pad_index)

    def collate_text_batch(self, batch, pad_index):
        data = batch
        lens = [len(x) for x in batch]
        return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_index), torch.tensor(lens)

    def get_test_size(self):
        return len(self.test_passwords)

    def get_train_size(self):
        return len(self.train_passwords)

    def remove_padding(self, password):
        # bos eos and pad tokens are already removed in tokenizer.decode_passwords
        return password

    def decode_password(self, password):
        # VGPT2 decodes passwords after the sampling.
        return password