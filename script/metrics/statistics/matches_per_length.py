import os

from collections import defaultdict
from script.utils.file_operations import read_files
from script.metrics.statistics.evaluator import Evaluator


def _compute_match_per_length(test_passwords, matches_set):
    diz_matches_total = defaultdict(lambda: [0, 0])

    for password in test_passwords:
        if not password:
            continue
        length = str(len(password))
        diz_matches_total[length][1] += 1
        if password in matches_set:
            diz_matches_total[length][0] += 1

    stats = {}

    for length in diz_matches_total:
        percentage = round((diz_matches_total[length][0] / diz_matches_total[length][1]) * 100, 2)
        stats[length] = [diz_matches_total[length][1], diz_matches_total[length][0], percentage]

    return stats


class RQ5_2Evaluator(Evaluator):
    def __init__(self, test_settings, search_settings, csv_settings):
        super().__init__(test_settings, search_settings, csv_settings)

    def _get_entries(self):
        searching_for = {}
        setting_strings = self._prepare_settings_strings()
        models = self.test_settings["models"]
        datasets = self.test_settings["train_datasets"]

        for model in models:
            for dataset in datasets:
                for setting_string in setting_strings:
                    test_settings, n_samples = setting_string.split(os.sep)
                    query = {
                        'model': model,
                        'train-dataset': dataset,
                        'test-settings': test_settings,
                        'n_samples': n_samples,
                    }
                    key = tuple(query.items())
                    searching_for.setdefault(key, 0)

        return self._search_entries(searching_for)


    def _compute_metrics(self, matches_paths, real_paths):
        for model in matches_paths:
            for setting_string in matches_paths[model]:

                if setting_string not in real_paths:
                    continue

                for dataset in matches_paths[model][setting_string]:
                    if dataset not in real_paths[setting_string]:
                        continue

                    matches_path = matches_paths[model][setting_string][dataset]
                    matches_set = set(read_files(matches_path))

                    real_path = real_paths[setting_string][dataset]
                    test_passwords = set(read_files(real_path))

                    stats = _compute_match_per_length(test_passwords, matches_set)
                    test_settings, n_samples = setting_string.split(os.sep)

                    variable_data = []
                    for key in stats:
                        variable_data.append([key, stats[key][0], stats[key][1], stats[key][2]])

                    path, rows = self.prepare_to_csv(model, dataset, test_settings, n_samples, variable_data)

                    if path not in self.written_rows:
                        self.written_rows[path] = []
                    for row in rows:
                        self.written_rows[path].append(row)


def main(test_settings):
    search_settings = {
        'mode': "matches",
        'real_data_mode': "test",
    }

    csv_settings = {
        'test_name': 'rq5.2',
        'fieldnames': ["model", "train-dataset", "test-settings", "n_samples", "length", "test-size", "matches",
                       "match_percentage"],
    }

    evaluator = RQ5_2Evaluator(test_settings, search_settings, csv_settings)
    return evaluator.written_rows