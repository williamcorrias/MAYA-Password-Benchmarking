import numpy as np
import sys
import os
import random
from tqdm import tqdm

sys.path.append(os.getcwd())

from script.config.config import read_config
from script.metrics.utils.common_op import read_passwords, parse_args
from script.metrics.IMD.lib.msid import msid_score
from script.metrics.utils.tokenizer import Tokenizer

CONFIG = "script/metrics/IMD/CONF/config.yaml"

def prepare_data(generated_passwords_path, real_passwords_path, max_length, tokenizer):
    random.seed(42)

    generated_passwords = read_passwords(generated_passwords_path, max_length, real_psw=False)
    generated_passwords = [password for password in generated_passwords]
    random.shuffle(generated_passwords)
    generated_passwords = np.array([tuple(tokenizer.encode_data(password)) for password in generated_passwords])

    real_passwords = read_passwords(real_passwords_path, max_length, real_psw=True)
    real_passwords = [password for password in real_passwords]
    random.shuffle(real_passwords)
    real_passwords = np.array([tuple(tokenizer.encode_data(password)) for password in real_passwords])

    return generated_passwords, real_passwords

def main():
    args = parse_args()
    params = read_config(CONFIG)

    char_bag = params["data"]["char_bag"]
    max_length = params["data"]["max_length"]
    runs = int(params["test"]["runs"])
    batch_size = int(params["test"]["batch_size"])

    tokenizer = Tokenizer(char_bag, max_length)

    generated_passwords_path = args.path_to_generated_passwords
    real_passwords_path = args.path_to_real_passwords
    name = args.name
    info = args.info

    print("Name: {}".format(name))
    if info is not None:
        print("Info: {}".format(info))

    generated_passwords, real_passwords = prepare_data(generated_passwords_path, real_passwords_path, max_length, tokenizer)

    scores = []

    progress_bar = tqdm(range(runs))

    k1 = min(len(generated_passwords), batch_size)
    k2 = min(len(real_passwords), batch_size)

    for run in range(runs):
        p_idx = np.random.choice(generated_passwords.shape[0], k1, replace=False)
        p = generated_passwords[p_idx]

        q_idx = np.random.choice(real_passwords.shape[0], k2, replace=False)
        q = real_passwords[q_idx]

        score = msid_score(p, q)
        scores.append(score)
        progress_bar.update(1)

        print(f'Run {run + 1}: {score}')

    print('IMD(gen_data, real_data)', np.average(scores))


if __name__ == '__main__':
    main()
