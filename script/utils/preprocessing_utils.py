import pickle


def train_test_split(data, train_split_percentage):
    assert 0 < train_split_percentage <= 100, "train_split_percentage must be between 0 and 100"
    split = int(len(data) * (float(train_split_percentage) / float(100.00)))
    train_passwords = data[0:split]
    test_passwords = data[split:]
    return train_passwords, test_passwords

