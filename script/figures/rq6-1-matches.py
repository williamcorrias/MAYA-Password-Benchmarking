import csv

from script.figures.various_plot import heatmap_table
def read_csv(path):
    data = {}

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            combo = row['Combo']
            if combo not in data:
                data[combo] = []
            mergeability = row['Mergeability']
            data[combo].append(float(mergeability))

        for combo in data:
            data[combo] = round(sum(data[combo]) / len(data[combo]), 5)

    return data


data = read_csv('script/figures/src/rq6-1-matches.csv')
models = ["FLA", "PassFlow", "PassGAN", "PassGPT", "PLR-GAN", "VGPT2"]

for model in models:
    data[f"{model.lower()}_{model.lower()}"] = 0

heatmap_table(data,
              models,
              vmin=0,
              vmax=1,
              dest_path=f"script/figures/output/rq6-1/mergeability_table.pdf",
              cbar_kws={"label": "Mergeability Index"}
)