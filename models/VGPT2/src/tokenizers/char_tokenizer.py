import pickle
import torch
import re

char = str

class CharTokenizer():

    def __init__(
        self,
        data_path,
        add_sos_and_eos: bool = True,
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):

        self.characters = self.find_chars(data_path)
        self.char_to_index = self.init_char_to_index_mapping(self.characters)

        self.add_sos_and_eos = add_sos_and_eos

        self.sos_token = sos_token
        self.sos_index = len(self.char_to_index)
        self.char_to_index[self.sos_token] = self.sos_index
        self.eos_token = eos_token
        self.eos_index = len(self.char_to_index)
        self.char_to_index[self.eos_token] = self.eos_index
        self.pad_token = pad_token
        self.pad_index = len(self.char_to_index)
        self.char_to_index[self.pad_token] = self.pad_index
        self.unk_token = unk_token
        self.unk_index = len(self.char_to_index)

        self.index_to_char = self.init_index_to_char_mapping(self.char_to_index)

        self.vocab_size = len(self.char_to_index)

    def find_chars(self, data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        data = set(data.split("\n"))
        unique_chars = set(''.join(set(char for password in data for char in password)))
        return unique_chars

    def init_char_to_index_mapping(self, characters: str):
        return {c: i for i, c in enumerate(characters)}

    def init_index_to_char_mapping(self, char_to_index_mapping):
        return list(char_to_index_mapping.keys())

    def get_vocab(self):
        return self.index_to_char

    def encode(self, text):
        indices = [self.char_to_index.get(c, self.unk_index) for c in list(text)]
        if self.add_sos_and_eos:
            indices = [self.sos_index] + indices + [self.eos_index]
        return torch.tensor(indices)

    def decode(self, indices):
        chars = [
            self.index_to_char[index]
            for index in indices
            if index not in [self.sos_index, self.eos_index, self.pad_index]
        ]
        return "".join(chars)