import csv
from script.figures.various_plot import multiline_graph


def read_csv(path):
    data = {}

    models = []

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = row['dataset']
            current_models = sorted(row['models'].replace(" ", "").split(","))
            if len(current_models) < 2:
                skip_model = current_models[0]
                continue

            if current_models not in models:
                models.append(current_models)

            if dataset not in data:
                data[dataset] = []

            match_percentage = float(row['delta'].replace("%", "").strip())
            data[dataset].append(match_percentage)

        for dataset in data:
            data[dataset] = sorted(data[dataset])

        data["Average"] = []
        for i in range(len(data[dataset])):
            avg = []
            for ds in data:
                if ds == "Average":
                    continue
                avg.append(data[ds][i])
            avg = sum(avg) / len(avg)
            data["Average"].append(avg)

        final_models = []
        seen = set()
        models.sort(key=lambda x: len(x))
        for combo in models:
            counter = len(combo)
            for model in combo:
                if model == skip_model:
                    continue
                if model not in seen:
                    final_models.append(f"$n_{counter}: {model}$")
                    seen.add(model)

    return data, final_models


data, models = read_csv('script/figures/src/rq6-2.csv')

models.insert(0, "$n_1: {FLA}$")

x_data = models
labels = []
y_data = []
for dataset in data:
    labels.append(dataset)
    to_append = [0]
    to_append.extend(data[dataset])
    y_data.append(to_append)

multiline_graph(x_data=x_data,
                y_data=y_data,
                x_caption="combinations of models",
                y_caption="gain (percentage points)",
                labels=labels,
                dest_path=f"script/figures/output/rq6-2/image.pdf",
                fontsize=11,
                labelsize=9,
)
