from script.plotters.plotter import Plotter

class RQ5_3Plotter(Plotter):
    def __init__(self, row, settings):
        super().__init__(row, settings, "rq5.3", weights=True)

    def _prepare_plot_settings(self):
        self.x_data = ["r" + str(n) for n in range(1, 20)]

    def _process_single_row(self, row):
        model = row["model"]
        train_dataset = row["train-dataset"]
        pattern = row['pattern']
        test_size = int(row['test-size'])
        n_samples = int(row["n_samples"])

        key = row['test-settings'] + f"-{n_samples}"
        if pattern in self.x_data:
            self.weights.setdefault(key, {}).setdefault(pattern, {}).setdefault(train_dataset, test_size)
            self.data.setdefault(key, {}).setdefault(model, {}).setdefault(train_dataset, {}).setdefault(pattern, None)
            self.data[key][model][train_dataset][pattern] = float(row["match_percentage"].replace("%", ""))

    def _sort_data(self):
        for key in self.data:
            self.data[key] = dict(sorted(self.data[key].items()))

    def _compute_weights(self):
        for test_settings in self.weights:
            for pattern in self.weights[test_settings]:
                total = 0
                for dataset in self.weights[test_settings][pattern]:
                    if self.weights[test_settings][pattern][dataset] is None:
                        self.weights[test_settings][pattern][dataset] = 0
                        continue
                    total += self.weights[test_settings][pattern][dataset]
                for dataset in self.weights[test_settings][pattern]:
                    self.weights[test_settings][pattern][dataset] = float(self.weights[test_settings][pattern][dataset]) / total
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

        header = " & "
        strings = []
        for test_settings in self.data:
            print(f"Settings: {test_settings}\n")
            for model in self.data[test_settings]:
                stringa = f"{model} &"
                for idx, regex in enumerate(self.data[test_settings][model]):
                    if idx == len(self.data[test_settings][model]) - 1:
                        stringa += str(round(self.data[test_settings][model][regex], 2)) + "\% \\\ "
                    else:
                        stringa += str(round(self.data[test_settings][model][regex], 2)) + "\% & "

                strings.append(stringa)

            for idx, regex in enumerate(self.x_data):
                if idx == len(self.x_data) - 1:
                    header += f"{regex} \\\ "
                else:
                    header += f"{regex} &"

            print(header)
            for x in strings:
                print(x)
            print("-" * 200)

    def _extra(self):
        pass

def main(rows=None, settings=None):
    RQ5_3Plotter(rows, settings)