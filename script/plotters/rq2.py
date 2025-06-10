import os
import csv

from script.plotters.various_plot import multiline_graph
from script.plotters.plotter import Plotter

class RQ2Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq2", weights=False)

    def _prepare_plot_settings(self):
        self.x_data = self.settings["n_samples"]

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]
        n_samples = int(row["n_samples"])

        key = row['test-settings']
        self.data.setdefault(key, {}).setdefault(train_dataset, {}).setdefault(model, {}).setdefault(n_samples, None)
        self.data[key][train_dataset][model][n_samples] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))
            for dataset in self.data[key]:
                self.data[key][dataset] = dict(sorted(self.data[key][dataset].items()))

    def _read_csv(self, path):
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self._process_single_row(row)

    def _plot_data(self):
        self._read_csv('script/plotters/src/rq2-traditional.csv')

        values = [self.data[test_settings][dataset][model][n_samples] for test_settings in self.data for dataset
                  in self.data[test_settings] for model in self.data[test_settings][dataset] for n_samples in self.data[test_settings][dataset][model]]

        if not values:
            print("No data to plot.")
            return

        max_y = max(values)

        x_ticks = ([1e6, 1e7, 1e8, 5e8], ['10^6', '10^7', '10^8', '5*10^8'])

        for test_settings in self.data:
            for dataset in self.data[test_settings]:
                labels = list(self.data[test_settings][dataset].keys())
                y_data = [[self.data[test_settings][dataset][model][n_samples] for n_samples in self.x_data] for model in labels]

                multiline_graph(x_data=self.x_data,
                                y_data=y_data,
                                x_caption="# generated passwords",
                                y_caption="% of guessed passwords",
                                x_log_scale=True,
                                x_lim=[min(self.x_data), max(self.x_data)],
                                y_lim=[-2, max_y + 1],
                                x_ticks=x_ticks,
                                labels=labels,
                                dest_path=os.path.join(self.dest_folder, f"{test_settings}-{dataset}.pdf"),
                                fontsize=13)

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ2Plotter(rows, settings)