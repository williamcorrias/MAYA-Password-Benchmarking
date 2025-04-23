import csv
from script.figures.various_plot import multiline_graph

train_sizes = [850000, 1785681, 2700000, 4710736, 11802325, 20000000, 40000000]
def read_csv(path):
    data = {}

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            model = row["model"]
            train_dataset = row['train-dataset']
            max_length = row['test-settings'].split("-")[1]
            train_size = row['test-settings'].split("-")[2]
            n_gen = row['#gen']

            if max_length != '12' or int(n_gen) != 500000000:
                continue

            if model not in data:
                data[model] = {}

            if train_dataset not in data[model]:
                data[model][train_dataset] = [None for _ in range(len(train_sizes))]

            index_to_place = train_sizes.index(int(train_size))
            match_percentage = row['match_percentage']

            data[model][train_dataset][index_to_place] = float(match_percentage.replace("%",""))

    for dataset in data:
        data[dataset] = dict(sorted(data[dataset].items()))

    return data

data = read_csv('script/figures/src/rq3.csv')

x_data = [i for i in range(len(train_sizes))]
x_ticks = (x_data, ['1e6', '2e6', '3e6', '5e6', '1e7', '2e7', '4e7'])

for model in data:
    labels = []
    y_data = []
    for dataset in data[model]:
        labels.append(dataset)
        percentages = data[model][dataset]
        y_data.append(data[model][dataset])

    multiline_graph(x_data=x_data,
                    y_data=y_data,
                    x_caption="train set sizes",
                    y_caption="% of guessed passwords",
                    x_log_scale=False,
                    x_ticks=x_ticks,
                    labels=labels,
                    dest_path=f"script/figures/output/rq3/{model}.pdf",
                    fontsize=13)