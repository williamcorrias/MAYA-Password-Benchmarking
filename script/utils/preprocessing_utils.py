import pickle
from script.utils.file_operations import load_pickle

class SkipCombinationException(Exception):
    pass

def train_test_split(data, train_split_percentage):
    assert 0 < train_split_percentage <= 100, "train_split_percentage must be between 0 and 100"
    split = int(len(data) * (float(train_split_percentage) / float(100.00)))
    train_passwords = data[0:split]
    test_passwords = data[split:]
    return train_passwords, test_passwords

def read_datasets(paths):
    data = []
    for path in paths:
        tmp_data = []
        for password in load_pickle(path):

            if password.endswith('\n'):
                password = password[:-1]

            if not all(32 <= ord(c) <= 127 for c in password):
                continue

            tmp_data.append(password)

        data.extend(tmp_data)
    return data
