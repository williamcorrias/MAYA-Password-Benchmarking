import csv
from script.figures.various_plot import multiline_graph

thresholds = [1000000, 2500000, 5000000, 7500000, 10000000, 25000000, 50000000, 75000000, 100000000, 250000000, 500000000]

def read_csv(path):
    data = {}
    weights = {}

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            model = row["model"]
            train_dataset = row['train-dataset']
            max_length = row['test-settings'].split("-")[1]

            if max_length not in weights:
                weights[max_length] = {}
            if train_dataset not in weights[max_length]:
                weights[max_length][train_dataset] = row["test-size"]

            if max_length not in data:
                data[max_length] = {}
            if model not in data[max_length]:
                data[max_length][model] = {}
            if train_dataset not in data[max_length][model]:
                data[max_length][model][train_dataset] = [None for _ in range(len(thresholds))]

            generated_password = row['#gen']
            index_to_place = thresholds.index(int(generated_password))
            match_percentage = row['match_percentage']

            data[max_length][model][train_dataset][index_to_place] = match_percentage

    for length in data:
        data[length] = dict(sorted(data[length].items()))

    return data, weights

def compute_weights(weights):
    for length in weights:
        total = sum(int(weights[length][dataset]) for dataset in weights[length])
        for dataset in weights[length]:
            weights[length][dataset] = int(weights[length][dataset]) / total

    return weights


def compute_weighted_average(data, weights):
    weights = compute_weights(weights)
    for length in data:
        for model in data[length]:
            avg = []
            index = 0
            somma = 0
            while index < len(thresholds):
                for dataset in data[length][model]:
                    somma += float(data[length][model][dataset][index].replace("%","")) * weights[length][dataset]
                avg.append(somma)
                somma = 0
                index += 1
            data[length][model] = avg
    return data

def marginal_gain(data):
    marginal_gain_wrt_total_matches = {}
    marginal_gain_wrt_previous_matches = {}

    for length in data:
        for model in data[length]:
            if model not in marginal_gain_wrt_total_matches:
                marginal_gain_wrt_total_matches[model] = {}
                marginal_gain_wrt_previous_matches[model] = {}

            for i, threshold in enumerate(range(len(thresholds) - 1)):
                lower_bound = thresholds[threshold]
                upper_bound = thresholds[threshold + 1]
                if f"{lower_bound} -> {upper_bound}" not in marginal_gain_wrt_total_matches[model]:
                    marginal_gain_wrt_total_matches[model][f"{lower_bound} -> {upper_bound}"] = []
                    marginal_gain_wrt_previous_matches[model][f"{lower_bound} -> {upper_bound}"] = []

                wrt_total_matches = (data[length][model][i+1] - data[length][model][i])
                wrt_previous_matches = ((data[length][model][i+1] - data[length][model][i]) / data[length][model][i]) * 100

                marginal_gain_wrt_total_matches[model][f"{lower_bound} -> {upper_bound}"].append(wrt_total_matches)
                marginal_gain_wrt_previous_matches[model][f"{lower_bound} -> {upper_bound}"].append(wrt_previous_matches)


    for model in marginal_gain_wrt_total_matches:
        for i, threshold in enumerate(range(len(thresholds) - 1)):
            lower_bound = thresholds[threshold]
            upper_bound = thresholds[threshold + 1]
            marginal_gain_wrt_total_matches[model][f"{lower_bound} -> {upper_bound}"] = round((
                    sum(marginal_gain_wrt_total_matches[model][f"{lower_bound} -> {upper_bound}"]) /
                    len(marginal_gain_wrt_total_matches[model][f"{lower_bound} -> {upper_bound}"])),2)

            marginal_gain_wrt_previous_matches[model][f"{lower_bound} -> {upper_bound}"] = round((
                    sum(marginal_gain_wrt_previous_matches[model][f"{lower_bound} -> {upper_bound}"]) /
                    len(marginal_gain_wrt_previous_matches[model][f"{lower_bound} -> {upper_bound}"])),2)

    return marginal_gain_wrt_total_matches, marginal_gain_wrt_previous_matches

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


data, weights = read_csv('script/figures/src/rq1.csv')
data = compute_weighted_average(data, weights)

x_data = thresholds
x_ticks = ([1e6, 1e7, 1e8, 5e8],
           ['10^6', '10^7','10^8','5*10^8'])

values = [value for length in data for model in data[length] for value in data[length][model]]
min_y = min(values)
max_y = max(values)

for length in data:
    labels = []
    y_data = []
    for model in data[length]:
        labels.append(model)
        y_data.append(data[length][model])

    multiline_graph(x_data=x_data,
                    y_data=y_data,
                    x_caption="# generated passwords",
                    y_caption="% of guessed passwords",
                    x_log_scale=True,
                    x_lim=[min(x_data), max(x_data)],
                    y_lim=[min_y, max_y+1],
                    x_ticks=x_ticks,
                    labels=labels,
                    dest_path=f"script/figures/output/rq1/length{length}.pdf",
                    fontsize=13,
                    legend_params={
                        "loc": "upper left",
                    })


mg_dict_wrt_total_matches, mg_dict_wrt_previous_matches = marginal_gain(data)
diz_to_csv(mg_dict_wrt_total_matches, "script/figures/output/rq1/marginal_gain_wrt_total_matches.csv")
diz_to_csv(mg_dict_wrt_previous_matches, "script/figures/output/rq1/marginal_gain_wrt_previous_matches.csv")
