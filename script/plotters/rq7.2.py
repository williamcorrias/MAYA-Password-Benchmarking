import os
from collections import defaultdict

from script.plotters.various_plot import multiline_graph
from script.plotters.plotter import Plotter

class RQ7_2Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq7.2", weights=True)

    def _prepare_plot_settings(self):
        self.x_data =  ["1-5", "6", "7", "8", "9", "10", "11", "12"]
        self.flatten_x_data = list(range(1, 13))

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]

        length = int(row['length'])
        n_samples = int(row["n_samples"])

        key = row['test-settings'] + f"-{n_samples}"
        if length in self.flatten_x_data:
            self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(length, None)
            self.data[key][model][train_dataset][length] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _format_data(self):
        for test_settings in self.data:
            formatted_data = defaultdict(lambda: defaultdict(dict))

            for model in self.data[test_settings]:
                for dataset in self.data[test_settings][model]:
                    grouped = defaultdict(list)
                    for length in self.data[test_settings][model][dataset]:
                        if int(length) <= 5:
                            grouped["1-5"].append(self.data[test_settings][model][dataset][length])
                        else:
                            grouped[str(length)].append(self.data[test_settings][model][dataset][length])

                    for length in grouped:
                        formatted_data[model][dataset][str(length)] = sum(grouped[str(length)])
            self.data[test_settings] = formatted_data

    def _compute_average(self):
        for test_settings in self.data:
            stats = defaultdict(dict)

            for model in self.data[test_settings]:
                for length in self.x_data:
                    total = 0
                    count = 0
                    for dataset in self.data[test_settings][model]:
                        if length in self.data[test_settings][model][dataset]:
                            total += self.data[test_settings][model][dataset][length]
                            count += 1
                    if count > 0:
                        stats[model][length] = round(total / count, 2)

            sorted_stats = {
                k: stats[k]
                for k in sorted(stats.keys())
                if k != 'real'
            }

            if 'real' in stats:
                sorted_stats['real'] = stats['real']

            self.data[test_settings] = sorted_stats

    def _plot_data(self):
        self._format_data()
        self._compute_average()

        for test_settings in self.data:
            labels = list(self.data[test_settings].keys())
            y_data = [[self.data[test_settings][model][str(x)] for x in self.x_data] for model in labels]

            multiline_graph(x_data=self.x_data,
                            y_data=y_data,
                            labels=labels,
                            x_caption="Password Lengths",
                            y_caption="Frequencies (%)",
                            dest_path=os.path.join(self.dest_folder, f"{test_settings}.pdf"))

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ7_2Plotter(rows, settings)