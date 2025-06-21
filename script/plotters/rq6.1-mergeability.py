import os

from script.plotters.various_plot import heatmap_table
from script.plotters.plotter import Plotter

class RQ6_1_MergeabilityPlotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq6.1-mergeability", weights=True)

    def _prepare_plot_settings(self):
        pass

    def _process_single_row(self, row):
        combo = row['combo']
        model1, model2 = combo.split("-")
        train_dataset = row["train-dataset"]

        n_samples = int(row["n_samples"])

        key = row['test-settings'] + f"-{n_samples}"
        self_key = f"{model1}-{model2}"
        self.data.setdefault(key, {}).setdefault(self_key, {}).setdefault(train_dataset, None)
        self.data[key][self_key][train_dataset] = float(row['mergeability-index'])
        for model in (model1, model2):
            self_key = f"{model}-{model}"
            self.data.setdefault(key, {}).setdefault(self_key, {}).setdefault(train_dataset, None)
            if self.data[key][self_key][train_dataset] is None:
                self.data[key][self_key][train_dataset] = 0

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _compute_average(self):
        for test_settings in self.data:
            for combo in self.data[test_settings]:
                sum = 0
                for dataset in self.data[test_settings][combo]:
                    sum += self.data[test_settings][combo][dataset]
                self.data[test_settings][combo] = round(sum / len(self.data[test_settings][combo]), 5)

    def _plot_data(self):
        self._compute_average()

        for test_settings in self.data:
            models = set()
            for combo in self.data[test_settings]:
                model1, model2 = combo.split("-")
                models.add(model1)
                models.add(model2)

            heatmap_table(self.data[test_settings],
                          sorted(models),
                          vmin=0,
                          vmax=1,
                          dest_path=os.path.join(self.dest_folder, f"{test_settings}-mergeability.pdf"),
                          cbar_kws={"label": "Mergeability Index"}
                          )

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ6_1_MergeabilityPlotter(rows, settings)