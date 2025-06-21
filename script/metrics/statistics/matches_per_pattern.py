import os
import re

from script.utils.file_operations import read_files
from script.metrics.statistics.evaluator import Evaluator

regex = {
    'r1': r'^[A-Za-z]+$',
    'r2': r'^[a-z]+$',
    'r3': r'^[A-Z]+$',
    'r4': r'^[0-9]+$',
    'r5': r'^[\W_]+$',
    'r6': r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]+$',
    'r7': r'^(?=.*[A-Za-z])(?=.*[\W_])[A-Za-z\W_]+$',
    'r8': r'^(?=.*\d)(?=.*[\W_])[\d\W_]+$',
    'r9': r'^(?=.*\d)(?=.*[\W_])(?=.*[A-Za-z])[A-Za-z\d\W_]+$',
    'r10': r'^[a-zA-Z][a-zA-Z0-9\W_]+[0-9]$',
    'r11': r'^[A-Za-z][A-Za-z0-9\W_]+[\W_]$',
    'r12': r'^[0-9][A-Za-z]+$',
    'r13': r'^[0-9][A-Za-z0-9\W_]+[\W_]$',
    'r14': r'^[0-9][A-Za-z0-9\W_]+[0-9]$',
    'r15': r'^[\W_][A-Za-z]+$',
    'r16': r'^[\W_][A-Za-z0-9\W_]+[\W_]$',
    'r17': r'^[\W_][A-Za-z0-9\W_]+[0-9]$',
    'r18': r'^[a-zA-Z0-9\W_]+[!]$',
    'r19': r'^[a-zA-Z0-9\W_]+[1]$',
}

def _compute_match_per_pattern(test_passwords, guessed_set):
    diz_matches_total = {}
    for pattern in regex:
        diz_matches_total[pattern] = [0, 0]

    for password in test_passwords:
        if not password:
            continue

        is_match = password in guessed_set

        for id, pattern in regex.items():
            if re.fullmatch(pattern, password):
                diz_matches_total[id][1] += 1
                if is_match:
                    diz_matches_total[id][0] += 1

    stats = {}

    for pattern in diz_matches_total:
        if diz_matches_total[pattern][1] != 0:
            percentage = round((diz_matches_total[pattern][0] / diz_matches_total[pattern][1]) * 100, 2)
            stats[pattern] = [diz_matches_total[pattern][1], diz_matches_total[pattern][0], percentage]
        else:
            stats[pattern] = [0, 0, 0]

    return stats

class RQ5_3Evaluator(Evaluator):

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

                    stats = _compute_match_per_pattern(test_passwords, matches_set)
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
        'test_name': 'rq5.3',
        'fieldnames': ["model", "train-dataset", "test-settings", "n_samples", "pattern", "test-size", "matches",
                      "match_percentage"]
    }

    evaluator = RQ5_3Evaluator(test_settings, search_settings, csv_settings)
    return evaluator.written_rows