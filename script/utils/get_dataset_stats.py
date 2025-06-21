import os
import glob
import gzip
import pickle
import pandas as pd

DATASET_FOLDER = "./datasets/"

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


def compute_length_distribution(name, path):
    diz_len = {
        "1-5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0, "12": 0, "13+": 0
    }

    data = read_files(path)

    total_passwords = len(data)

    for password in data:
        length = len(password)

        if 5 < length < 13:
            length = str(length)
        else:
            if length <= 5:
                length = "1-5"
            else:
                length = "13+"

        diz_len[str(length)] += 1

    for key in diz_len:
        diz_len[key] = round(diz_len[key] * 100 / total_passwords, 2)

    sorted_dict = dict(diz_len.items())

    print(f"Name: {name}")
    print(f"Length Distribution: {sorted_dict}")
    return sorted_dict

def get_dataset_path(dataset):
    pattern = os.path.join(DATASET_FOLDER, '**', dataset+'.pickle')
    dataset = glob.glob(pattern, recursive=True)
    return dataset

def to_latex(data, datasets):
    df = pd.DataFrame(data).round(2)
    order_dict = {name: idx for idx, name in enumerate(datasets)}
    df['order'] = df['dataset'].map(order_dict)
    df = df.sort_values(by='order').drop(columns='order')
    latex_table = df.to_latex(index=False, float_format="{:0.2f}".format)
    print(latex_table)

def main():
    datasets = ["rockyou", "linkedin", "mailru", "000webhost", "taobao", "gmail", "ashleymadison", "libero"]
    table = {"dataset": [], "1-5": [], "6": [], "7": [], "8": [], "9": [], "10": [], "11": [], "12": [], "13+": []}

    for dataset in datasets:
        path = get_dataset_path(dataset)
        assert path != [], f"Dataset {dataset} not found"
        path = path[0]
        diz = compute_length_distribution(dataset, path)

        table["dataset"].append(dataset)
        for key in diz:
            table[key].append(diz[key])

    table["dataset"].append("Average")
    table["dataset"].append("CDF")
    cdf = 0
    for key in table:
        if key == "dataset":
            continue
        avg = sum(table[key]) / len(table[key])
        cdf += avg
        table[key].append(avg)
        table[key].append(cdf)

    to_latex(table, datasets)


if __name__ == "__main__":
    main()
