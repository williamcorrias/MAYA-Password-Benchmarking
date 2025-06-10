import os
import csv

from script.plotters.various_plot import multiline_graph
from script.plotters.plotter import Plotter


class RQ1Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq1", weights=True)

    def _prepare_plot_settings(self):
        self.x_data = self.settings["n_samples"]

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]
        n_samples = int(row["n_samples"])

        key = row['test-settings']
        self.weights.setdefault(key, {}).setdefault(train_dataset, int(row["test-size"]))
        self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(n_samples, 0)
        self.data[key][model][train_dataset][n_samples] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _compute_weights(self):
        for test_settings, datasets in self.weights.items():
            total = sum(int(count) for count in datasets.values())
            for dataset in datasets:
                if self.weights[test_settings][dataset] is None:
                    self.weights[test_settings][dataset] = 0
                    continue
                self.weights[test_settings][dataset] = float(self.weights[test_settings][dataset]) / total
        return self.weights

    def _compute_weighted_average(self):
        self._compute_weights()

        for test_settings, models in self.data.items():
            for model, datasets in models.items():
                avg = []
                for i in sorted(self.x_data):
                    weighted_sum = 0.0
                    for dataset in datasets:
                        value = datasets[dataset][i]
                        if isinstance(value, str) and value.endswith("%"):
                            value = float(value.replace("%", ""))
                        else:
                            value = float(value)
                        weighted_sum += value * self.weights[test_settings][dataset]
                    avg.append(weighted_sum)

                self.data[test_settings][model] = avg

    def _plot_data(self):
        self._compute_weighted_average()
        values = [v for test_settings in self.data for model in self.data[test_settings]
                  for v in self.data[test_settings][model]]
        if not values:
            print("No data to plot.")
            return
        min_y, max_y = min(values), max(values)

        x_ticks = ([1e6, 1e7, 1e8, 5e8], ['10^6', '10^7', '10^8', '5*10^8'])

        for test_settings in self.data:
            labels = list(self.data[test_settings].keys())
            y_data = [self.data[test_settings][model] for model in labels]

            multiline_graph(x_data=self.x_data,
                            y_data=y_data,
                            x_caption="# generated passwords",
                            y_caption="% of guessed passwords",
                            x_log_scale=True,
                            x_lim=[min(self.x_data), max(self.x_data)],
                            y_lim=[min_y, max_y + 1],
                            x_ticks=x_ticks,
                            labels=labels,
                            dest_path=os.path.join(self.dest_folder, f"{test_settings}.pdf"),
                            fontsize=13,
                            legend_params={
                                "loc": "upper left",
                            })

    def _extra(self):
        if not self.data:
            return
        total_gain, relative_gain = self._compute_marginal_gain()
        self._diz_to_csv(total_gain, os.path.join(self.dest_folder, "marginal_gain_absolute.csv"))
        self._diz_to_csv(relative_gain, os.path.join(self.dest_folder, "marginal_gain_relative.csv"))

    def _compute_marginal_gain(self,):
        gain_total, gain_relative = {}, {}

        for test_settings, models in self.data.items():
            for model, values in models.items():
                gt = gain_total.setdefault(model, {} )
                gr = gain_relative.setdefault(model, {} )

                for i in range(len(self.x_data) - 1):
                    lower = self.x_data[i]
                    upper = self.x_data[i + 1]
                    label = f"{lower} -> {upper}"

                    gt.setdefault(label, [])
                    gr.setdefault(label, [])

                    current = values[i + 1]
                    previous = values[i]

                    total_gain = current - previous

                    try:
                        relative_gain = (total_gain / previous) * 100
                    except ZeroDivisionError:
                        relative_gain = 0.0

                    gt[label].append(total_gain)
                    gr[label].append(relative_gain)

        for model in gain_total:
            for i in range(len(self.x_data) - 1):
                label = f"{self.x_data[i]} -> {self.x_data[i + 1]}"
                gt_values = gain_total[model][label]
                gr_values = gain_relative[model][label]

                gain_total[model][label] = round(sum(gt_values) / len(gt_values), 2) if gt_values else 0.0
                gain_relative[model][label] = round(sum(gr_values) / len(gr_values), 2) if gr_values else 0.0

        return gain_total, gain_relative

    def _diz_to_csv(self, data, output_file):
        header = ["Model"] + list(next(iter(data.values())).keys())

        with open(output_file, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

            for model, x_data in data.items():
                row = [model] + list(x_data.values())
                writer.writerow(row)

def main(rows=None, settings=None):
    RQ1Plotter(rows, settings)