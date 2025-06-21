import os

from script.plotters.various_plot import bar_graph
from script.plotters.plotter import Plotter

class RQ5_2Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq5.2", weights=True)

    def _prepare_plot_settings(self):
        self.x_data = [x for x in range(4, 13)]

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]
        length = int(row['length'])
        test_size = int(row['test-size'])
        n_samples = int(row["n_samples"])

        key = row['test-settings'] + f"-{n_samples}"
        if length in self.x_data:
            self.weights.setdefault(key, {}).setdefault(length, {}).setdefault(train_dataset, test_size)
            self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(length, None)
            self.data[key][model][train_dataset][length] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _compute_weights(self):
        for test_settings in self.weights:
            for length in self.weights[test_settings]:
                total = 0
                for dataset in self.weights[test_settings][length]:
                    if self.weights[test_settings][length][dataset] is None:
                        self.weights[test_settings][length][dataset] = 0
                        continue
                    total += self.weights[test_settings][length][dataset]
                for dataset in self.weights[test_settings][length]:
                    self.weights[test_settings][length][dataset] = float(self.weights[test_settings][length][dataset]) / total
        return self.weights

    def _compute_weighted_average(self):
        self._compute_weights()
        output_data = {}

        for test_settings, models in self.data.items():
            for model, datasets in models.items():
                if test_settings not in output_data:
                    output_data[test_settings] = {}

                if model not in output_data[test_settings]:
                    output_data[test_settings][model] = {}

                for x in self.x_data:
                    if x not in output_data[test_settings][model]:
                        output_data[test_settings][model][x] = 0.0
                    weighted_sum = 0.0

                    for dataset, values in datasets.items():
                        value = values[x]
                        if isinstance(value, str) and value.endswith("%"):
                            value = float(value.replace("%", ""))
                        else:
                            value = float(value)

                        weighted_sum += value * self.weights[test_settings][x][dataset]

                    output_data[test_settings][model][x] = weighted_sum

        self.data = output_data


    def _plot_data(self):
        self._compute_weighted_average()

        x_ticks = (self.x_data, self.x_data)

        for test_settings in self.data:
            labels = list(self.data[test_settings].keys())
            y_data = [[self.data[test_settings][model][x] for x in self.x_data] for model in labels]

            bar_graph(x_data=self.x_data,
                      y_data=y_data,
                      x_caption="test-set password length",
                      y_caption="% of guessed passwords",
                      x_ticks=x_ticks,
                      labels=labels,
                      dest_path=os.path.join(self.dest_folder, f"{test_settings}.pdf"),
                      bar_width=1.15,
                      fontsize=26,
                      labelsize=24,
                      legend_params={"fontsize": 24},
                      fig_size=(20, 10),
                      )

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ5_2Plotter(rows, settings)