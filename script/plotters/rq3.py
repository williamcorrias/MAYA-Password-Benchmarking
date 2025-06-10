import os

from script.plotters.various_plot import multiline_graph
from script.plotters.plotter import Plotter

class RQ3Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq3", weights=False)

    def _prepare_plot_settings(self):
        self.x_data = self.settings["train_chunk_percentage"]

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]
        train_size = int(row['test-settings'].split("-")[2])
        n_samples = int(row['n_samples'])

        key = row['test-settings']
        parts = key.split("-")
        parts.append(str(n_samples))
        key = "-".join([p for i, p in enumerate(parts) if i != 2])
        if train_size in self.x_data:
            self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(train_size, None)
            self.data[key][model][train_dataset][train_size] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            for model in self.data[key]:
                self.data[key][model] = dict(sorted(self.data[key][model].items()))

    def _plot_data(self):
        x_data = list(range(len(self.x_data)))

        for test_settings in self.data:
            for model in self.data[test_settings]:
                for dataset in self.data[test_settings][model]:
                    tmp_list = []
                    for value in sorted(self.x_data):
                        if value in self.data[test_settings][model][dataset]:
                            tmp_list.append(self.data[test_settings][model][dataset][value])
                        else:
                            tmp_list.append(None)
                    self.data[test_settings][model][dataset] = tmp_list

        x_ticks = (x_data, ['1e6', '2e6', '3e6', '5e6', '1e7', '2e7', '4e7'])

        for test_settings in self.data:
            for model in self.data[test_settings]:
                labels = list(self.data[test_settings][model].keys())
                y_data = [self.data[test_settings][model][train_dataset] for train_dataset in labels]

                multiline_graph(x_data=x_data,
                                y_data=y_data,
                                x_caption="train set sizes",
                                y_caption="% of guessed passwords",
                                x_log_scale=False,
                                x_ticks=x_ticks,
                                labels=labels,
                                dest_path=os.path.join(self.dest_folder, f"{test_settings}-{model}.pdf"),
                                fontsize=13)

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ3Plotter(rows, settings)