import random
import sys
import os

sys.path.append(os.getcwd())

from script.config.config import read_config
from script.metrics.utils.tokenizer import Tokenizer
from script.metrics.utils.common_op import read_passwords, parse_args
from script.metrics.psw_classifier.classifier import PasswordClassifier

CONFIG = "script/metrics/psw_classifier/CONF/config.yaml"

def prepare_data(generated_passwords_path, real_passwords_path, max_length, split_percentage):
    random.seed(42)

    generated_passwords = read_passwords(generated_passwords_path, max_length, real_psw=False)
    random.shuffle(generated_passwords)

    real_passwords = read_passwords(real_passwords_path, max_length, real_psw=True)
    random.shuffle(real_passwords)

    generated_passwords_train, generated_passwords_test = split_dataset(generated_passwords, split_percentage)
    real_passwords_train, real_passwords_test = split_dataset(real_passwords, split_percentage)

    train_dataset = (generated_passwords_train, real_passwords_train)
    test_dataset = (generated_passwords_test, real_passwords_test)

    return train_dataset, test_dataset


def split_dataset(dataset, split_percentage):
    split_index = int(len(dataset) * split_percentage / 100)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]
    return train_dataset, test_dataset


def main():
    args = parse_args()
    params = read_config(CONFIG)

    char_bag = params["data"]["char_bag"]
    max_length = params["data"]["max_length"]
    train_test_percentage = params["data"]["train_test_percentage"]

    generated_passwords_path = args.path_to_generated_passwords
    real_passwords_path = args.path_to_real_passwords
    name = args.name
    info = args.info

    print("Name: {}".format(name))
    if info is not None:
        print("Info: {}".format(info))

    train_dataset, test_dataset = prepare_data(generated_passwords_path, real_passwords_path, max_length,
                                               train_test_percentage)

    tokenizer = Tokenizer(char_bag, max_length)

    model = PasswordClassifier(train_dataset, test_dataset, tokenizer, name, params)

    try:
        print("Loading Checkpoint...")
        model.load(args.name + ".pt")
        print("Checkpoint Loaded Successfully!")
    except:
        print("Training Model...")
        model.train()
        print("Model Trained Successfully!")

    print("Computing Divergence: ")

    avg_loss = model.eval()


if __name__ == '__main__':
    main()
