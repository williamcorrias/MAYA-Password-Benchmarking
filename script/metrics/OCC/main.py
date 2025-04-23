import matplotlib.pyplot as plt
import random
import sys
import os

sys.path.append(os.getcwd())

from script.config.config import read_config
from script.metrics.utils.common_op import read_passwords, parse_args
from script.metrics.OCC.occ import OCC
from script.metrics.utils.tokenizer import Tokenizer

CONFIG = "script/metrics/OCC/CONF/config.yaml"

def prepare_data(generated_passwords_path, real_passwords_path, max_length, tokenizer):
    random.seed(42)

    generated_passwords = read_passwords(generated_passwords_path, max_length, real_psw=False)
    random.shuffle(generated_passwords)
    generated_passwords = [tuple(tokenizer.encode_data(password)) for password in generated_passwords]

    real_passwords = read_passwords(real_passwords_path, max_length, real_psw=True)
    random.shuffle(real_passwords)
    real_passwords = [tuple(tokenizer.encode_data(password)) for password in real_passwords]

    return generated_passwords, real_passwords


def plot_graph_precision_coverage(alphas, alpha_precision_curve, beta_coverage_curve):
    plt.plot(alphas, alpha_precision_curve, color="b")
    plt.plot(alphas, beta_coverage_curve, color="r")
    plt.plot(alphas, alphas, color="black", linestyle="--")

    plt.show()

def main():
    args = parse_args()
    params = read_config(CONFIG)

    max_length = params['data']['max_length']
    char_bag = params['data']['char_bag']

    tokenizer = Tokenizer(char_bag, max_length)

    generated_passwords_path = args.path_to_generated_passwords
    real_passwords_path = args.path_to_real_passwords
    name = args.name
    info = args.info

    print("Name: {}".format(name))
    if info is not None:
        print("Info: {}".format(info))

    generated_passwords, real_passwords = prepare_data(generated_passwords_path, real_passwords_path, max_length,
                                                       tokenizer)

    data = (generated_passwords, real_passwords)

    model = OCC(data, params, name)

    try:
        print("Loading Checkpoint...")
        model.load(args.name + ".pt")
        print("Checkpoint Loaded Successfully!")
    except:
        print("Training Model...")
        model.train()
        print("Model Trained Successfully!")

    alphas, alpha_precision_curve, delta_alpha_precision, beta_recall_curve, delta_beta_recall, authen = model.eval()

    plot_graph_precision_coverage(alphas, alpha_precision_curve, beta_recall_curve)

    print(f"Delta precision alpha: {delta_alpha_precision}")
    print(f"Delta coverage beta: {delta_beta_recall}")
    print(f"Authen: {authen}")


if __name__ == '__main__':
    main()
