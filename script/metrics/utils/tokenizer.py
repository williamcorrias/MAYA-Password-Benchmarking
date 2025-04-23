class Tokenizer:

    def __init__(self, char_bag, max_length):
        self.char_bag = char_bag

        self.char_indices = dict((c, i) for i, c in enumerate(self.char_bag))
        self.indices_char = dict((i, c) for i, c in enumerate(self.char_bag))

        self.pad_token = str("<PAD>")
        self.char_indices[self.pad_token] = len(char_bag)
        self.indices_char[len(char_bag)] = self.pad_token

        self.vocab_size = len(self.char_bag) + 1

        self.max_length = max_length

    def encode_data(self, data):
        encoded_data = [self.char_indices[char] for char in data]
        if len(encoded_data) < self.max_length:
            encoded_data = self.pad_data(encoded_data)
        return encoded_data

    def pad_data(self, data):
        while len(data) < self.max_length:
            data.append(self.char_indices[self.pad_token])
        return data[:self.max_length]