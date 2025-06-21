import os
from collections import defaultdict

from script.plotters.various_plot import multiline_graph
from script.plotters.plotter import Plotter

class RQ6_2Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq6.2", weights=False)

    def _prepare_plot_settings(self):
        self.x_data = {}
        self.models_name = {
            'fla': 'FLA',
            'passgan': 'PassGAN',
            'plrgan': 'PLR',
            'passflow': 'PassFlow',
            'passgpt': 'PassGPT',
            'vgpt2': 'VGPT2',
        }

    def _process_single_row(self, row):
        combo = row['combo']
        train_dataset = row['train-dataset']

        n_samples = int(row['n_samples'])

        key = row['test-settings'] + f"-{n_samples}"
        self.data.setdefault(key, {}).setdefault(combo, {}).setdefault(train_dataset, None)
        self.data[key][combo][train_dataset] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _compute_delta(self):
        new_data = defaultdict(list)

        for test_settings in self.data:
            for combo, results in self.data[test_settings].items():
                for dataset, value in results.items():
                    new_data[dataset].append(value)

            for dataset, values in new_data.items():
                values.sort()
                new_data[dataset] = [round(values[i] - values[0], 2) for i in range(1, len(values))]

            num_points = len(next(iter(new_data.values())))
            n_datasets = len(new_data)
            new_data["Average"] = [
                round(sum(new_data[ds][i] for ds in new_data if ds != "Average") / n_datasets, 2)
                for i in range(num_points)
            ]

            seen = set()
            final_models = []
            for combo in sorted(self.data[test_settings].keys(), key=lambda x: len(x.split("-"))):
                count = len(combo.split("-"))
                for model in combo.split("-"):
                    if model not in seen:
                        seen.add(model)
                        label = self.models_name.get(model, model)
                        final_models.append(f"$n_{{{count}}}: {label}$")

            self.data[test_settings] = new_data
            if test_settings not in self.x_data:
                self.x_data[test_settings] = []
            self.x_data[test_settings] = final_models

    def _plot_data(self):
        self._compute_delta()

        for test_settings in self.data:
            labels = list(self.data[test_settings].keys())
            y_data = [[0] + self.data[test_settings][label] for label in labels]

            multiline_graph(x_data=self.x_data[test_settings],
                            y_data=y_data,
                            x_caption="combinations of models",
                            y_caption="gain (percentage points)",
                            labels=labels,
                            dest_path=os.path.join(self.dest_folder, f"{test_settings}.pdf"),
                            fontsize=10,
                            labelsize=9,
                            )

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ6_2Plotter(rows, settings)