import csv
from script.figures.various_plot import bar_graph

thresholds = [x for x in range(4,13)]
def read_csv(path):
    data = {}
    weights = {}

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            if not row:
                continue

            model = row["model"]
            train_dataset = row['dataset']
            key = row['key']
            match_percentage = row['percentage']
            test_size = row['test-size']

            if int(key) not in thresholds:
                continue

            if key not in weights:
                weights[key] = {}

            if train_dataset not in weights[key]:
                weights[key][train_dataset] = test_size

            if model not in data:
                data[model] = {}

            if train_dataset not in data[model]:
                data[model][train_dataset] = {}

            if key not in data[model][train_dataset]:
                data[model][train_dataset][key] = match_percentage

    return data, weights

def compute_weights(weights):
    for key in weights:
        total = sum(float(weights[key][dataset]) for dataset in weights[key])
        for dataset in weights[key]:
            weights[key][dataset] = float(weights[key][dataset]) / total
    return weights

def compute_weighted_average(data, weights):
    output_data = {}
    weights = compute_weights(weights)
    for model in data:
        if model not in output_data:
            output_data[model] = {}

        for index in range(len(thresholds)):
            somma = 0
            key = str(thresholds[index])
            if key not in output_data[model]:
                output_data[model][key] = 0.0

            for dataset in data[model]:
                if key in data[model][dataset]:
                    somma += float(data[model][dataset][key].replace("%","")) * weights[key][dataset]

            output_data[model][key] = somma

    return output_data

data, weights = read_csv('script/figures/src/match_per_length.csv')
data = compute_weighted_average(data, weights)

x_data = list(range(len(thresholds)))
x_ticks = (x_data, ["4", "5", "6", "7", "8", "9", "10", "11", "12"])

y_data = []
labels = []

data = dict(sorted(data.items()))

for model in data:
    labels.append(model)
    percentages = data[model]
    y_data.append(data[model].values())

length_avg_gen = {
    'passgan': {'4': 0.95, '5': 3.22, '6': 16.46, '7': 17.25, '8': 25.49, '9': 14.64, '10': 12.47, '11': 6.26, '12': 2.98},
    'plr_gan': {'4': 0.39, '5': 1.38, '6': 11.81, '7': 13.47, '8': 25.71, '9': 17.07, '10': 15.67, '11': 8.46, '12': 5.98},
    'passgpt': {'4': 0.42, '5': 1.43, '6': 10.59, '7': 13.19, '8': 24.93, '9': 17.38, '10': 16.18, '11': 9.14, '12': 6.57},
    'passflow': {'4': 0.45, '5': 3.6, '6': 23.49, '7': 20.86, '8': 27.44, '9': 10.59, '10': 7.34, '11': 3.85, '12': 2.15},
    'fla': {'4': 0.16, '5': 1.08, '6': 11.49, '7': 15.97, '8': 34.0, '9': 17.62, '10': 12.48, '11': 4.55, '12': 2.64},
    'vgpt2': {'4': 0.06, '5': 0.59, '6': 6.45, '7': 12.19, '8': 22.93, '9': 26.52, '10': 18.45, '11': 8.6, '12': 4.15}
}

point_dict = {}

for idx, model in enumerate(labels):
    point_dict[str(idx)] = []
    for key in thresholds:
        point_dict[str(idx)].append(length_avg_gen[model][str(key)])

bar_graph(x_data=x_data,
          y_data=y_data,
          x_caption="test-set password length",
          y_caption="% of guessed passwords",
          x_ticks=x_ticks,
          labels=labels,
          dest_path=f"script/figures/output/rq5-length/match_per_length.pdf",
          bar_width=1.15,
          fontsize=26,
          labelsize=24,
          legend_params={"fontsize": 24},
          fig_size=(20, 10),
          point_dict=point_dict,
          )