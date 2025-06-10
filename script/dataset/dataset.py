import collections
import os
import pickle
import random
import numpy as np

SAVE_FOLDER = "./data/dataset"
if not os.path.exists(os.path.join(os.getcwd(), SAVE_FOLDER)):
    os.makedirs(os.path.join(os.getcwd(), SAVE_FOLDER), exist_ok=True)

class Dataset:
    def __init__(self, train_path, test_path, max_length, name, max_train_size=0, max_test_size=0, skip_unk=False):
        self.train_path = train_path
        self.test_path = test_path
        self.max_length = int(max_length)
        self.max_train_size = max_train_size
        self.max_test_size = max_test_size
        self.skip_unk = skip_unk

        self.name = name

        self.train_passwords = list()
        self.test_passwords = set()
        self.charmap = {}
        self.inv_charmap = []

        if not self.load():
            self.load_dataset(is_train=True)
            self.load_dataset(is_train=False)

        self.charmap_size = len(self.charmap)
        self.train_passwords_size = len(self.train_passwords)
        self.test_passwords_size = len(self.test_passwords)

        print(f'train {self.train_passwords_size} test {self.test_passwords_size}')
        self.save()

    def get_train_size(self):
        return len(self.train_passwords)

    def get_test_size(self):
        return len(self.test_passwords)

    def get_charmap_size(self):
        return self.charmap_size

    def load(self):
        full_path = os.path.join(SAVE_FOLDER, self.name + ".pickle")
        if os.path.exists(full_path):
            print(f'Loading dataset {full_path} from pickle.')
            with open(full_path, 'rb') as fin:
                loaded = pickle.load(fin)
            if loaded['max_length'] == self.max_length and loaded['max_train_size'] == self.max_train_size \
                    and loaded['max_test_size'] == self.max_test_size:
                self.__dict__.update(loaded)
                print(f'Loaded dataset {full_path} from pickle.')
                return True

            print(f'{full_path} saved on disk has different parameters from current run. Rebuilding')
        return False

    def save(self):
        full_path = os.path.join(SAVE_FOLDER, self.name + ".pickle")
        if not os.path.exists(full_path):
            with open(full_path, 'wb') as fout:
                pickle.dump(self.__dict__, fout)
                print('Pickled dataset saved')

    def load_dataset(self, max_vocab_size=2048, is_train=True):
        lines = []
        with open(self.train_path if is_train else self.test_path, 'rb') as f:
            data = pickle.load(f)

        data = data.split("\n")

        for line in data:
            line = tuple(line)

            if not line:
                continue

            # right pad with ` character
            lines.append(self.pad_password(line))

        if self.max_train_size != 0 and is_train:
            # keep seed fixed through different runs
            prng = random.Random(42)
            lines = prng.sample(lines, self.max_train_size)

        np.random.shuffle(lines)
        counts = collections.Counter(char for line in lines for char in line)

        if is_train:
            self.charmap = {}
            self.inv_charmap = []
            if not self.skip_unk:
                self.charmap['unk'] = 0
                self.inv_charmap.append('unk')

            for char, count in counts.most_common(max_vocab_size - 1):
                if char not in self.charmap:
                    self.charmap[char] = len(self.inv_charmap)
                    self.inv_charmap.append(char)

        passwords = []
        for line in lines:
            filtered_line = []
            for char in line:
                if char in self.charmap:
                    filtered_line.append(char)
                else:   # this condition should never be triggered
                    if self.skip_unk:
                        filtered_line = None
                        break
                    filtered_line.append('unk')

            if filtered_line is not None:
                passwords.append(tuple(filtered_line))
        if is_train:
            self.train_passwords = [tuple(self.encode_password(pwd)) for pwd in passwords]
        else:
            self.test_passwords = set([tuple(self.encode_password(pwd)) for pwd in passwords])
            if self.max_test_size and self.max_test_size < len(self.test_passwords):
                self.test_passwords = set(random.sample(set([tuple(self.encode_password(pwd)) for pwd in passwords]),
                                                        self.max_test_size))
                passwords = self.test_passwords

        print('{} set: loaded {} out of {} lines in dataset. {} filtered'
              .format('Training' if is_train else 'Test', len(passwords), len(lines), len(lines) - len(passwords)))

    def encode_password(self, padded_password):
        return [self.charmap[c] for c in padded_password]

    def pad_password(self, password):
        return password + (("`",) * (self.max_length - len(password)))

    def decode_password(self, encoded_password):
        return tuple([self.inv_charmap[c] if 0 <= c < self.charmap_size else c for c in encoded_password])

    def get_batches(self, batch_size=128, is_train=True):
        data = self.train_passwords if is_train else self.test_passwords

        np.random.shuffle(data)

        for i in range(0, len(data) - batch_size + 1, batch_size):
            yield np.array([np.array(pwd) for pwd in data[i:i + batch_size]], dtype='float32')