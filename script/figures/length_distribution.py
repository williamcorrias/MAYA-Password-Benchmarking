import pickle
import gzip
import os

from various_plot import multiline_graph

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
            data = [psw.strip() for psw in data]

    return data

def find_guesses(results_path):
    data = {}
    models = os.listdir(results_path)
    for model in models:
        if not os.path.isdir(os.path.join(results_path, model)):
            continue

        data[model] = []

        model_path = os.path.join(results_path, model)
        datasets = os.listdir(model_path)
        for dataset in datasets:
            if dataset == "yandex":
                continue
            dataset_path = os.path.join(model_path, dataset)
            settings_dir = os.listdir(dataset_path)
            for setting in settings_dir:
                if setting != "all-12-100-80":
                    continue
                setting_path = os.path.join(dataset_path, setting, "500000000")
                hash = os.listdir(setting_path)[0]
                guesses_path = os.path.join(setting_path, hash, "guesses")
                file = os.listdir(guesses_path)[0]
                final_path = os.path.join(guesses_path, file)
                data[model].append(final_path)
    return data

def compute_length_distribution(path):
    diz_len = {
        "1-5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0, "12": 0,
    }

    data = read_files(path)

    total_passwords = len(data)

    for password in data:
        length = len(password)

        if length > 12:
            continue

        if length <= 5:
            length = "1-5"

        diz_len[str(length)] += 1

    for key in diz_len:
        diz_len[key] = round(diz_len[key] * 100 / total_passwords, 2)

    sorted_dict = dict(diz_len.items())

    return sorted_dict

paths = find_guesses("results/test_chunked_data_training")

data = {}

for model in paths:
    if model not in data:
        data[model] = {}
    for path in paths[model]:
        length_distr = compute_length_distribution(path)
        for range in length_distr:
            if range not in data[model]:
                data[model][range] = []
            data[model][range].append(length_distr[range])

    for range in data[model]:
        data[model][range] = round(sum(data[model][range]) / len(data[model][range]), 2)

print(data)

data = dict(sorted(data.items()))
data = {k : [v for v in vals.values()] for k,vals in data.items()}

data['REAL_AVG'] = [2.80, 17.57, 14.70, 25.94, 15.31, 12.92, 6.39, 4.32]

y_data = []
labels = []

for model in data:
    labels.append(model)
    y_data.append(data[model])

x_data = ["1-5", "6", "7", "8", "9", "10", "11", "12"]

multiline_graph(x_data=x_data,
                y_data=y_data,
                labels=labels,
                x_caption="Password Lengths",
                y_caption="Frequencies (%)",
                dest_path="script/figures/output/rq7/length_distribution.pdf",
                )
