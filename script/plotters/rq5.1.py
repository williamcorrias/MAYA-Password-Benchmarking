import os
import numpy as np

from script.plotters.various_plot import bar_graph
from script.plotters.plotter import Plotter

class RQ5_1Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq5.1", weights=True)

    def _prepare_plot_settings(self):
        self.x_data = self.settings["test_frequency"]

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]
        n_samples = int(row["n_samples"])

        key = row['test-settings']
        parts = key.split("-")
        parts.append(str(n_samples))

        frequency = parts[2]
        try:
            frequency = int(frequency)
            key = "-".join([p for i, p in enumerate(parts) if i != 2])
        except ValueError:
            frequency = -int(parts[3])
            key = "-".join([p for i, p in enumerate(parts) if i != 3 and p])

        if train_dataset != "linkedin" and train_dataset != "ashleymadison":
            if frequency in self.x_data:
                self.weights.setdefault(key, {}).setdefault(frequency, {}).setdefault(train_dataset, int(row["test-size"]))

                self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(frequency, None)
                self.data[key][model][train_dataset][frequency] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _compute_weights(self):
        for test_settings in self.weights:
            for frequency in self.weights[test_settings]:
                total = 0
                for dataset in self.weights[test_settings][frequency]:
                    if self.weights[test_settings][frequency][dataset] is None:
                        self.weights[test_settings][frequency][dataset] = 0
                        continue
                    total += self.weights[test_settings][frequency][dataset]
                for dataset in self.weights[test_settings][frequency]:
                    self.weights[test_settings][frequency][dataset] = float(self.weights[test_settings][frequency][dataset]) / total
        return self.weights

    def _compute_weighted_average(self):
        y_errors = {}
        output_data = {}
        self._compute_weights()

        for test_settings, models in self.data.items():
            for model, datasets in models.items():
                if test_settings not in output_data:
                    output_data[test_settings] = {}
                    y_errors[test_settings] = {}

                if model not in output_data:
                    output_data[test_settings][model] = [None for _ in range(len(self.x_data))]
                    y_errors[test_settings][model] = [None for _ in range(len(self.x_data))]

                for i in range(len(self.x_data)):
                    weighted_sum = 0.0
                    frequency = self.x_data[i]
                    all_values = []
                    for dataset, values in datasets.items():
                        value = values[frequency]
                        if isinstance(value, str) and value.endswith("%"):
                            value = float(value.replace("%", ""))
                        else:
                            value = float(value)
                        weighted_sum += value * self.weights[test_settings][frequency][dataset]
                        all_values.append(value)

                    output_data[test_settings][model][i] = weighted_sum
                    std_dev = np.std(np.array(all_values))
                    y_errors[test_settings][model][i] = std_dev
        return output_data, y_errors


    def _plot_data(self):
        data, std_dev = self._compute_weighted_average()

        self.data = data

        x_data = list(range(len(self.x_data)))
        x_ticks = (x_data, ["top 5%", "top 10%", "bottom 90%"])

        for test_settings in self.data:

            labels = list(self.data[test_settings].keys())
            y_data = [self.data[test_settings][model] for model in labels]
            y_errors = [std_dev[test_settings][model] for model in labels]

            bar_graph(x_data=x_data,
                      y_data=y_data,
                      y_errors=y_errors,
                      x_caption="test-set password frequency (%)",
                      y_caption="% of guessed passwords",
                      x_ticks=x_ticks,
                      labels=labels,
                      dest_path=os.path.join(self.dest_folder, f"{test_settings}.pdf"),
                      bar_width=0.3,
                      fontsize=13,
                      )

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ5_1Plotter(rows, settings)