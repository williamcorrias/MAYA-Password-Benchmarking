import gzip
import pickle

from script.figures.various_plot import multiline_graph

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


def compute_length_distribution(path, x_data):
    data = read_files(path)
    diz_len = {}
    for x in x_data:
        diz_len[x] = 0

    for psw in data:
        if not psw:
            continue

        lun = len(psw)

        if lun in diz_len:
            diz_len[lun] += 1

    sorted_dict = dict(sorted(diz_len.items()))

    print(f"Length Distribution: {sorted_dict}")

    total_passwords = sum(sorted_dict.values())
    percentages = [round(values / total_passwords * 100, 2) for values in sorted_dict.values()]
    return percentages

file_paths = {
    "FLA": "results/test_chunked_data_training/fla/000webhost/all-12-100-80/10000000/432a04011c519801862411bc38aebb0e/guesses/Epoch2-most_prob_n_psw.gz",
    "PassGPT": "results/test_chunked_data_training/passgpt/000webhost/all-12-100-80/10000000/432a04011c519801862411bc38aebb0e/guesses/guesses.gz",
    "PassFlow": "results/test_chunked_data_training/passflow/000webhost/all-12-100-80/10000000/432a04011c519801862411bc38aebb0e/guesses/Epoch104.gz",
    "000webhost": "data/splitted/test-432a04011c519801862411bc38aebb0e.pickle",
}

data = []
labels = []
x_data = list(range(4,13))

for file in file_paths:
    y_data = compute_length_distribution(file_paths[file], x_data)
    data.append(y_data)

    labels.append(file)

multiline_graph(x_data=x_data,
                y_data=data,
                labels=labels,
                x_caption="Password Lengths",
                y_caption="Frequencies (%)",
                dest_path="script/figures/output/rq7/IMD-analysis.pdf",
)