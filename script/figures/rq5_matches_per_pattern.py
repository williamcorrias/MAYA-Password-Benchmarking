import csv

thresholds = ["r"+str(n) for n in range(1,20)]
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

            if key not in thresholds:
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

data, weights = read_csv('script/figures/src/match_per_pattern.csv')
data = compute_weighted_average(data, weights)

for regex in thresholds:
    stringa = str(regex) + " & "
    for model in data:
        stringa += str(round(data[model][regex], 2)) + "\% & "
    print(stringa)