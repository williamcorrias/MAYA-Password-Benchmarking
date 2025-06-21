import collections
import os
import pickle
import numpy as np

SAVE_FOLDER = "./data/dataset"
if not os.path.exists(os.path.join(os.getcwd(), SAVE_FOLDER)):
    os.makedirs(os.path.join(os.getcwd(), SAVE_FOLDER), exist_ok=True)

class Dataset:
    def __init__(self, train_passwords, test_passwords, max_length, name):
        self.name = name
        self.max_length = int(max_length)

        self.train_passwords = list(train_passwords)
        self.test_passwords = set(test_passwords)

        self.charmap = {}
        self.inv_charmap = []

        if not self.load():
            self.load_dataset(is_train=True)
            self.load_dataset(is_train=False)

        self.charmap_size = len(self.charmap)
        self.save()

        print(f'train {len(self.train_passwords)} test {len(self.test_passwords)}')

    def get_train_size(self):
        return len(self.train_passwords)

    def load(self):
        full_path = os.path.join(SAVE_FOLDER, self.name + ".pickle")
        if os.path.exists(full_path):
            print(f'Loading dataset {full_path} from pickle.')
            with open(full_path, 'rb') as fin:
                loaded = pickle.load(fin)
            if loaded['max_length'] == self.max_length:
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
        data = self.train_passwords if is_train else self.test_passwords

        for line in data:
            line = tuple(line)

            if not line:
                continue

            lines.append(self.pad_password(line))

        np.random.shuffle(lines)
        counts = collections.Counter(char for line in lines for char in line)

        if is_train:
            self.charmap = {}
            self.inv_charmap = []

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
                else:  # this condition should never be triggered
                    filtered_line = None
                    break

            if filtered_line is not None:
                passwords.append(tuple(filtered_line))

        if is_train:
            self.train_passwords = [tuple(self.encode_password(pwd)) for pwd in passwords]
        else:
            self.test_passwords = set([tuple(self.encode_password(pwd)) for pwd in passwords])

        print('{} set: loaded {} out of {} lines in dataset. {} filtered'
              .format('Training' if is_train else 'Test', len(passwords), len(lines), len(lines) - len(passwords)))

    def encode_password(self, password):
        return [self.charmap[c] for c in password]

    def decode_password(self, encoded_password):
        decoded_password = ''
        for c in encoded_password:
            if 0 <= c < self.charmap_size:
                decoded_password += self.inv_charmap[c]
            else:
                return None
        return decoded_password

    def pad_password(self, password):
        return password + (("`",) * (self.max_length - len(password)))

    def remove_padding(self, password):
        return password.replace('`', '')

    def get_batches(self, batch_size=128, is_train=True):
        data = self.train_passwords if is_train else self.test_passwords

        np.random.shuffle(data)

        for i in range(0, len(data) - batch_size + 1, batch_size):
            yield np.array([np.array(pwd) for pwd in data[i:i + batch_size]], dtype='float32')