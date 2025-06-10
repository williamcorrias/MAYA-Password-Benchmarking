from script.plotters.plotter import Plotter

class RQ4_1Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq4.1", weights=False)

    def _prepare_plot_settings(self):
        pass

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]
        settings = row["test-settings"].split("-")
        test_dataset = settings[2]
        n_samples = int(row['n_samples'])

        key = row['test-settings']
        parts = key.split("-")
        parts.append(str(n_samples))
        key = "-".join([p for i, p in enumerate(parts) if i != 2])

        self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(test_dataset, None)
        self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(train_dataset, None)
        self.data[key][model][train_dataset][test_dataset] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _plot_data(self):
        header = "\\multirow{2}{*}{}"
        sub_header = "Train / Test"

        for test_settings in self.data:
            print(f"Settings: {test_settings}\n")
            models = list(self.data[test_settings].keys())

            train_datasets = set()
            test_datasets = set()
            for model in self.data[test_settings]:
                train_datasets.update(self.data[test_settings][model].keys())
                for train in self.data[test_settings][model]:
                    test_datasets.update(self.data[test_settings][model][train].keys())

            train_datasets = sorted(train_datasets)
            test_datasets = sorted(test_datasets)

            for model in models:
                header += f" & \\multicolumn{{3}}{{c|}}{{{model}}}"
            header += " \\\\"

            for _ in models:
                for test in test_datasets:
                    sub_header += f" & {test}"
            sub_header += " \\\\"

            print("\\begin{tabular}{|c|" + "c|" * (3 * len(models)) + "}")
            print("\\hline")
            print(header)
            print("\\cline{2-" + str(1 + 3 * len(models)) + "}")
            print(sub_header)
            print("\\hline")

            for train in train_datasets:
                row = train
                for model in models:
                    for test in test_datasets:
                        if test in self.data[test_settings][model][train]:
                            value = self.data[test_settings][model][train][test]
                        else:
                            value = "N/A"
                        if isinstance(value, float):
                            value = f"{value:.2f}\\%"
                        row += f" & {value}"
                row += " \\\\"
                print(row)
                print("\\hline")

            print("\\end{tabular}")
            print("\n")


    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ4_1Plotter(rows, settings)