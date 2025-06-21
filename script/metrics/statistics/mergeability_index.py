import os

from itertools import combinations
from script.metrics.statistics.evaluator import Evaluator
from script.utils.file_operations import read_files

def _compute_mergeability(set1, set2):
    union = len(set1 | set2)
    matches = (len(set1), len(set2))
    mergeability_idx = (union - max(matches)) / max(matches)
    return mergeability_idx

class RQ6_1_MergeabilityEvaluator(Evaluator):
    def __init__(self, test_settings, search_settings, csv_settings):
        super().__init__(test_settings, search_settings, csv_settings)

    def _get_entries(self):
        searching_for = {}
        setting_strings = self._prepare_settings_strings()
        models = self.test_settings["models"]
        datasets = self.test_settings["train_datasets"]

        models = sorted(list(models))
        combos = sorted(list(combinations(models, 2)))

        for model1, model2 in combos:
            combo = f"{model1}-{model2}"
            for dataset in datasets:
                for setting_string in setting_strings:
                    test_settings, n_samples = setting_string.split(os.sep)
                    query = {
                        'combo': f"{combo}",
                        'train-dataset': dataset,
                        'test-settings': test_settings,
                        'n_samples': n_samples,
                    }
                    key = tuple(query.items())
                    searching_for.setdefault(key, 0)

        missing_entries = self._search_entries(searching_for)
        formatted_entries = []
        for entry in missing_entries:
            parts = entry.split(",")
            combo = parts[0]
            model1, model2 = combo.split("-")
            parts = ",".join(parts[1:])

            formatted_entries.append(f"{model1},{parts}")
            formatted_entries.append(f"{model2},{parts}")

        return formatted_entries

    def _compute_metrics(self, matches_paths, real_paths):
        models = sorted(list(matches_paths.keys()))
        combos = sorted(list(combinations(models, 2)))

        for model1, model2 in combos:
            combo = f"{model1}-{model2}"
            for setting_string in matches_paths[model1]:
                for dataset in matches_paths[model1][setting_string]:
                    if setting_string not in matches_paths[model1]:
                        continue
                    if dataset not in matches_paths[model2][setting_string]:
                        continue

                    model1_path = matches_paths[model1][setting_string][dataset]
                    model1_set = set(read_files(model1_path))

                    model2_path = matches_paths[model2][setting_string][dataset]
                    model2_set = set(read_files(model2_path))

                    stats = _compute_mergeability(model1_set, model2_set)

                    test_settings, n_samples = setting_string.split(os.sep)
                    variable_data = [[stats]]
                    path, rows = self.prepare_to_csv(combo, dataset, test_settings, n_samples, variable_data)

                    if path not in self.written_rows:
                        self.written_rows[path] = []
                    for row in rows:
                        self.written_rows[path].append(row)


def main(test_settings):
    search_settings = {
        'mode': "matches",
        'real_data_mode': "",
    }

    csv_settings = {
        'test_name': 'rq6.1-mergeability',
        'fieldnames': ["combo", "train-dataset", "test-settings", "n_samples", "mergeability-index"]

    }

    evaluator = RQ6_1_MergeabilityEvaluator(test_settings, search_settings, csv_settings)
    return evaluator.written_rows