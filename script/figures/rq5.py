import csv
import numpy as np
from script.figures.various_plot import bar_graph

thresholds = [5, 10, -90]

def read_csv(path):
    data = {}
    weights = {}

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            if not row:
                continue

            model = row["model"]
            train_dataset = row['train-dataset']

            if train_dataset == "linkedin" or train_dataset == "ashleymadison":
                continue

            settings = row['test-settings'].split("-")

            frequency = settings[2]

            try:
                frequency = int(frequency)
            except ValueError:
                frequency = -int(settings[3])

            generated_password = int(row['#gen'])
            max_length = int(settings[1])

            if max_length != 12 or generated_password != 500000000:
                continue

            if str(frequency) not in weights:
                weights[str(frequency)] = {}

            if train_dataset not in weights[str(frequency)]:
                weights[str(frequency)][train_dataset] = int(row["test-size"])

            if model not in data:
                data[model] = {}

            if train_dataset not in data[model]:
                data[model][train_dataset] = [None for _ in range(len(thresholds))]

            match_percentage = row['match_percentage']
            index_to_place = thresholds.index(frequency)
            data[model][train_dataset][index_to_place] = match_percentage

    return data, weights

def compute_weights(weights):
    for frequency in weights:
        total = sum(weights[frequency][dataset] for dataset in weights[frequency])
        for dataset in weights[frequency]:
            weights[frequency][dataset] /= total
    return weights


def compute_weighted_average(data, weights):
    y_errors = {}
    output_data = {}
    weights = compute_weights(weights)
    for model in data:
        if model not in output_data:
            output_data[model] = [None for _ in range(len(thresholds))]
            y_errors[model] = [None for _ in range(len(thresholds))]

        for index in range(len(thresholds)):
            somma = 0
            frequency = str(thresholds[index])
            values = []
            for dataset in data[model]:
                somma += float(data[model][dataset][index].replace("%","")) * weights[frequency][dataset]
                values.append(float(data[model][dataset][index].replace("%","")))
            output_data[model][index] = somma
            std_dev = np.std(np.array(values))
            y_errors[model][index] = std_dev
    return output_data, y_errors

def diz_to_csv(data, output_file):
    with (open(output_file, mode="w", newline='') as file):
        writer = csv.writer(file)
        header = ["Model"] + list(next(iter(data.values())).keys())
        writer.writerow(header)

        for model in data:
            row = [model]
            for threshold in data[model].values():
                row.append(threshold)
            writer.writerow(row)


data, weights = read_csv('script/figures/src/rq5.csv')
data, std_dev = compute_weighted_average(data, weights)

x_data = list(range(len(thresholds)))
x_ticks = (x_data, ["top 5%", "top 10%", "bottom 90%"])

y_data = []
y_errors = []
labels = []

data = dict(sorted(data.items()))

for model in data:
    labels.append(model)
    percentages = data[model]
    y_data.append(data[model])
    y_errors.append(std_dev[model])

bar_graph(x_data=x_data,
        y_data=y_data,
        y_errors=y_errors,
        x_caption="test-set password frequency (%)",
        y_caption="% of guessed passwords",
        x_ticks=x_ticks,
        labels=labels,
        dest_path=f"script/figures/output/rq5/rq5.pdf",
        fontsize=13)