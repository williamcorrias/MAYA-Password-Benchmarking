import torch
import pickle
import re
import numpy as np

from models.VGPT2.src.tokenizers.char_tokenizer import CharTokenizer


class TokenizedTextDataLoader:
    def __init__(self, train_path, test_path, max_length, params):

        self.tokenizer = CharTokenizer(train_path)

        self.pad_index = self.tokenizer.pad_index
        self.sos_index = self.tokenizer.sos_index
        self.eos_index = self.tokenizer.eos_index
        self.unk_index = self.tokenizer.unk_index

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size = params["batch_size"]
        self.max_sequence_length = max_length
        self.max_size = params['max_size']

        decoded_train_passwords = self.create_dataset(train_path, is_train=True)
        self.train_passwords = [self.tokenizer.encode(data) for data in decoded_train_passwords]

        test_passwords = self.create_dataset(test_path, is_train=False)
        test_passwords = set(test_passwords)
        self.test_passwords = set(test_passwords) - set(self.train_passwords)

    def create_dataset(self, data_path, is_train):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        data = data.split("\n")

        if is_train and (self.max_size > 0):
            data = data[:self.max_size]

        return set(data)

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
