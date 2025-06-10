from collections import Counter
import random
import copy

from script.utils.preprocessing_utils import *

def read_train_passwords(train_passwords, test_passwords, **kwargs):
    train_paths = kwargs['train_datasets']
    train_passwords = read_datasets(train_paths)
    return train_passwords, test_passwords

def read_test_passwords(train_passwords, test_passwords, **kwargs):
    if 'test_datasets' in kwargs:
        test_paths = kwargs['test_datasets']
        test_passwords = read_datasets(test_paths)
    return train_passwords, test_passwords

def filter_by_length(train_passwords, test_passwords, **kwargs):
    max_len = int(kwargs['max_length'])

    if not (max_len > 0):
        raise SkipCombinationException("max_length must be greater than 0")

    train_passwords = [password for password in train_passwords if max_len >= len(password) > 0]
    test_passwords = [password for password in test_passwords if max_len >= len(password) > 0]

    return train_passwords, test_passwords


def filter_by_char_bag(train_passwords, test_passwords, **kwargs):
    char_bag = str(kwargs['char_bag'])
    train_passwords = [password for password in train_passwords if all(char in char_bag for char in password)]
    test_passwords = [password for password in test_passwords if all(char in char_bag for char in password)]

    return train_passwords, test_passwords


def standard_split(train_passwords, test_passwords, **kwargs):
    initial_test_passwords = copy.deepcopy(test_passwords)  # if cross dataset

    train_split_percentage = kwargs['train_split_percentage']
    train_passwords, test_passwords = train_test_split(train_passwords, train_split_percentage)

    test_passwords = test_passwords - set(train_passwords)

    if initial_test_passwords:  # if cross dataset
        test_passwords = initial_test_passwords  # use as a test other dataset passwords

    return train_passwords, test_passwords


def test_centric_split(train_passwords, test_passwords, **kwargs):
    initial_test_passwords = copy.deepcopy(test_passwords)  # if cross dataset
    random.shuffle(train_passwords)

    train_split_percentage = kwargs['train_split_percentage']
    train_passwords, test_passwords = train_test_split(train_passwords, train_split_percentage)

    train_counts = Counter(train_passwords)
    test_counts = Counter(set(test_passwords))

    set_train_passwords = [password for password, _ in train_counts.items() if test_counts.get(password, 0) == 0]

    train_passwords = []
    for password in set_train_passwords:
        count = train_counts.get(password)
        train_passwords.extend([password] * count)

    random.shuffle(train_passwords)

    if initial_test_passwords:  # if cross dataset
        test_passwords = initial_test_passwords  # use as a test other dataset passwords

    return train_passwords, test_passwords


def filter_by_frequency(train_passwords, test_passwords, **kwargs):
    test_frequency = int(kwargs['test_frequency'])

    if not (-100 <= test_frequency <= 100):
        raise SkipCombinationException("test_frequency must be between -100 and 100")

    counter = Counter(test_passwords)

    if test_frequency > 0:
        n = max(1, int(len(set(test_passwords)) * (test_frequency / 100.0)))
        test_passwords = [password for password, _ in counter.most_common(n)]
    else:
        n = max(1, int(len(set(test_passwords)) * ((test_frequency * -1) / 100.0)))
        sorted_passwords = counter.most_common()
        sorted_passwords.reverse()
        test_passwords = [password for password, _ in sorted_passwords[:n]]

    return train_passwords, test_passwords


def chunk_train_dataset(train_passwords, test_passwords, **kwargs):
    train_chunk_percentage = kwargs['train_chunk_percentage']
    if not (train_chunk_percentage > 0):
        raise SkipCombinationException("train_chunk_percentage must be greater than 0")

    if 100 < train_chunk_percentage < len(train_passwords):
        train_passwords = train_passwords[:train_chunk_percentage]
    elif 0 < train_chunk_percentage < 100:
        chunk_index = len(train_passwords) * train_chunk_percentage // 100
        train_passwords = train_passwords[:chunk_index]
    elif train_chunk_percentage - 1 == len(train_passwords):
        train_passwords = train_passwords
    elif train_chunk_percentage > len(train_passwords):
        raise SkipCombinationException("train_chunk_percentage is larger than dataset size.")

    return train_passwords, test_passwords

