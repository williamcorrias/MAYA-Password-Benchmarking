import csv
from script.figures.various_plot import multiline_graph

thresholds = [1000000, 2500000, 5000000, 7500000, 10000000, 25000000, 50000000, 75000000, 100000000, 250000000, 500000000]

def read_csv(path):
    data = {}

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            model = row["model"]
            train_dataset = row['train-dataset']
            max_length = row['test-settings'].split("-")[1]

            if max_length != '12':
                continue

            if train_dataset not in data:
                data[train_dataset] = {}
            if model not in data[train_dataset]:
                data[train_dataset][model] = [None for _ in range(len(thresholds))]

            generated_password = row['#gen']
            index_to_place = thresholds.index(int(generated_password))
            match_percentage = row['match_percentage']

            data[train_dataset][model][index_to_place] = float(match_percentage.replace("%",""))

    for dataset in data:
        data[dataset] = dict(sorted(data[dataset].items()))

    return data

data = read_csv('script/figures/src/rq2.csv')
data2 = read_csv('script/figures/src/rq2-jtr-hashcat.csv')

for dataset in data:
    data[dataset].update(data2[dataset])

x_data = thresholds
x_ticks = ([1e6, 1e7, 1e8, 5e8], ['10^6', '10^7','10^8','5*10^8'])

for dataset in data:
    labels = []
    y_data = []
    min_y = -1
    max_y = -1
    for model in data[dataset]:
        labels.append(model)
        percentages = data[dataset][model]

        if min(percentages) < min_y or min_y == -1:
            min_y = min(percentages)

        if max(percentages) > max_y or max_y == -1:
            max_y = max(percentages)

        y_data.append(data[dataset][model])

    multiline_graph(x_data=x_data,
                    y_data=y_data,
                    x_caption="# generated passwords",
                    y_caption="% of guessed passwords",
                    x_log_scale=True,
                    x_lim=[min(x_data), max(x_data)],
                    x_ticks=x_ticks,
                    labels=labels,
                    dest_path=f"script/figures/output/rq2/{dataset}.pdf",
                    fontsize=13)