import gzip
import pickle
import re

from script.figures.various_plot import multiline_graph

regex = {
    'r1': r'^[A-Za-z]+$',  #only letters.
    'r2': r'^[a-z]+$',  #only lower case letters.
    'r3': r'^[A-Z]+$',  #only upper case letters.
    'r4': r'^[0-9]+$',  #only numbers.
    'r5': r'^[\W_]+$',  #only special char.
    'r6': r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]+$',  #letters and numbers
    'r7': r'^(?=.*[A-Za-z])(?=.*[\W_])[A-Za-z\W_]+$',  #letters and special char
    'r8': r'^(?=.*\d)(?=.*[\W_])[\d\W_]+$',  #special char and numbers
    'r9': r'^(?=.*\d)(?=.*[\W_])(?=.*[A-Za-z])[A-Za-z\d\W_]+$',  #letters, special char and numbers
    'r10': r'^[a-zA-Z][a-zA-Z0-9\W_]+[0-9]$',  #starting with a letter and ending with a number
    'r11': r'^[A-Za-z][A-Za-z0-9\W_]+[\W_]$',  #starting with a letter and ending with a special char
    'r12': r'^[0-9][A-Za-z]+$',  #starting with a number and then letters
    'r13': r'^[0-9][A-Za-z0-9\W_]+[\W_]$',  #starting with a number and ending with a special char
    'r14': r'^[0-9][A-Za-z0-9\W_]+[0-9]$',  #starting with a number and ending with a number
    'r15': r'^[\W_][A-Za-z]+$',  #starting with a special char and then letters
    'r16': r'^[\W_][A-Za-z0-9\W_]+[\W_]$',  #starting with a special char and ending with a special char
    'r17': r'^[\W_][A-Za-z0-9\W_]+[0-9]$',  #starting with a special char and ending with a number
    'r18': r'^[a-zA-Z0-9\W_]+[!]$',  #ending with !
    'r19': r'^[a-zA-Z0-9\W_]+[1]$',  #ending with 1
}

def read_files(path):
    data = []

    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            data = f.read().split("\n")

    elif path.endswith('.txt'):
        with open(path, 'r') as f:
            data = f.read().split("\n")

    elif path.endswith('.pickle'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            data = data.split("\n")

    return data


def get_pattern_stats(path):
    patterns = {'r1': 0, 'r2': 0, 'r3': 0, 'r4': 0, 'r5': 0, 'r6': 0, 'r7': 0, 'r8': 0, 'r9': 0, 'r10': 0, 'r11': 0,
                'r12': 0, 'r13': 0, 'r14': 0, 'r15': 0, 'r16': 0, 'r17': 0, 'r18': 0, 'r19': 0}

    passwords = read_files(path)

    for password in passwords:
        for id, pattern in regex.items():
            if re.fullmatch(pattern, password):
                patterns[id] += 1

    for id, count in patterns.items():
        patterns[id] = round(count / len(passwords) * 100, 2)

    return patterns

file_paths = {
    "FLA": "results/test_chunked_data_training/fla/000webhost/all-12-100-80/10000000/432a04011c519801862411bc38aebb0e/guesses/Epoch2-most_prob_n_psw.gz",
    "PassGPT": "results/test_chunked_data_training/passgpt/000webhost/all-12-100-80/10000000/432a04011c519801862411bc38aebb0e/guesses/guesses.gz",
    "PassFlow": "results/test_chunked_data_training/passflow/000webhost/all-12-100-80/10000000/432a04011c519801862411bc38aebb0e/guesses/Epoch104.gz",
    "000webhost": "data/splitted/test-432a04011c519801862411bc38aebb0e.pickle",
    "Fully-Random-Psw": "script/metrics/utils/10million_fully_random_psw.txt"
}

data = []
labels = []
x_data = []

for file in file_paths:
    y_data = get_pattern_stats(file_paths[file])
    x_data = [key for key in y_data.keys()]

    data.append(y_data.values())

    labels.append(file)

multiline_graph(x_data=x_data,
                y_data=data,
                labels=labels,
                x_caption="Patterns",
                y_caption="Frequencies (%)",
                dest_path="script/figures/output/rq7/IMD-pattern-analysis.pdf",
)