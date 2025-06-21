import argparse
from script.utils.file_operations import read_files
import numpy as np
import random
random.seed(42)
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_generated_passwords',
                        type=str,
                        required=True,
                        nargs="+",
                        help='Path to generated passwords.')

    parser.add_argument('--path_to_real_passwords',
                        type=str,
                        required=True,
                        nargs="+",
                        help='Path to real passwords.')

    parser.add_argument('--name',
                        type=str,
                        required=False,
                        help='Name.')

    parser.add_argument('--info',
                        type=str,
                        required=False,
                        help='This string wil be printed during the evaluation for additional information.')

    return parser.parse_args()

def get_batches(dataset, batch_size):
    for i in range(0, len(dataset) - batch_size + 1, batch_size):
        yield dataset[i:i + batch_size]


def read_passwords(path_list, max_length, real_psw=True):
    output = []
    for file in path_list:
        data = read_files(file)
        data = [pw for pw in data if len(pw) <= max_length]
        if not real_psw and len(data) > 10**7:
            data_idx = random.sample(range(0, len(data)), 10**7)
            data = [data[idx] for idx in data_idx]
        output.extend(data)
    return output