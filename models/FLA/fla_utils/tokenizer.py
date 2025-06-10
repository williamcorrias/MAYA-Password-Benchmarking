import numpy as np

class Tokenizer():
    def __init__(self, chars, maxlen, PASSWORD_END, embedding=False, padding_character=False):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen
        self.vocab_size = len(self.chars)
        self.char_list = self.chars
        self.embedding = embedding
        self.padding_character = padding_character
        self.PASSWORD_END = PASSWORD_END

    def pad_to_len(self, astring, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        if len(astring) > maxlen:
            return astring[len(astring) - maxlen:]
        if self.padding_character:
            astring = astring + (self.PASSWORD_END * (maxlen - len(astring)))
            return astring
        return astring

    def encode_many(self, string_list, maxlen=None, y_vec=False):
        maxlen = maxlen if maxlen else self.maxlen
        x_str_list = map(lambda x: self.pad_to_len(x, maxlen), string_list)
        if self.embedding and not y_vec:
            x_vec = np.zeros(shape=(len(string_list), maxlen), dtype=np.int8)
        else:
            x_vec = np.zeros((len(string_list), maxlen, self.vocab_size), dtype=np.bool_)
        for i, xstr in enumerate(x_str_list):
            self.encode_into(x_vec[i], xstr)
        return x_vec

    def encode_many_chunks(self, string_list, max_input_str_len, maxlen=None, y_vec=False):
        maxlen = maxlen if maxlen else self.maxlen
        chunks_str_list = []
        iters = list(range(maxlen, max_input_str_len, maxlen // 2))
        iters.append(max_input_str_len)
        for a_string in string_list:
            prev_iter = 0
            for i in iters:
                if prev_iter >= len(a_string) and (len(a_string) != 0) or \
                        (len(a_string) == 0 and prev_iter != 0):
                    break
                chunk = a_string[i - maxlen:i]
                chunks_str_list.append(chunk)
                prev_iter = i

        return self.encode_many(chunks_str_list, maxlen, y_vec=y_vec), chunks_str_list

    def y_encode_into(self, Y, C):
        for i, c in enumerate(C):
            Y[i, self.char_indices[c]] = 1

    def encode_into(self, X, C):
        for i, c in enumerate(C):
            if len(X.shape) == 1:
                X[i] = self.char_indices[c]
            elif len(X.shape) == 2:
                X[i, self.char_indices[c]] = 1
            else:
                raise Exception("Code should never reach here, dimension of X can only be 1 or 2")

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        if self.embedding:
            X = np.zeros((maxlen), dtype=np.int8)
        else:
            X = np.zeros((maxlen, len(self.chars)))

        self.encode_into(X, C)
        return X

    def get_char_index(self, character):
        return self.char_indices[character]

    def translate(self, astring):
        return astring

