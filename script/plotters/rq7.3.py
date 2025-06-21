import os

from collections import defaultdict

from script.plotters.various_plot import multiline_graph
from script.plotters.plotter import Plotter

class RQ7_3Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq7.3", weights=True)

    def _prepare_plot_settings(self):
        self.x_data = ["r" + str(n) for n in range(1, 20)]

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]

        pattern = row['pattern']
        n_samples = int(row["n_samples"])

        key = row['test-settings'] + f"-{n_samples}"
        if pattern in self.x_data:
            self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(pattern, None)
            self.data[key][model][train_dataset][pattern] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _compute_average(self):
        for test_settings in self.data:
            stats = defaultdict(dict)

            for model in self.data[test_settings]:
                for pattern in self.x_data:
                    total = 0
                    count = 0
                    for dataset in self.data[test_settings][model]:
                        if pattern in self.data[test_settings][model][dataset]:
                            total += self.data[test_settings][model][dataset][pattern]
                            count += 1
                    if count > 0:
                        stats[model][pattern] = round(total / count, 2)

            sorted_stats = {
                k: stats[k]
                for k in sorted(stats.keys())
                if k != 'real'
            }

            if 'real' in stats:
                sorted_stats['real'] = stats['real']

            self.data[test_settings] = sorted_stats

    def _plot_data(self):
        self._compute_average()

        for test_settings in self.data:
            labels = list(self.data[test_settings].keys())
            y_data = [[self.data[test_settings][model][str(x)] for x in self.x_data] for model in labels]

            multiline_graph(x_data=self.x_data,
                            y_data=y_data,
                            labels=labels,
                            x_caption="Patterns",
                            y_caption="Frequencies (%)",
                            dest_path=os.path.join(self.dest_folder, f"{test_settings}.pdf"))

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ7_3Plotter(rows, settings)