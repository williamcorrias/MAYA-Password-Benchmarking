import os
import copy

from collections import defaultdict
from script.metrics.statistics.evaluator import Evaluator
from script.utils.file_operations import read_files

def multi_models_attack(matches_paths, test_paths):
    stats = defaultdict(lambda: defaultdict(dict))

    for settings_string in test_paths:
        datasets = test_paths[settings_string]

        test_passwords = {
            ds: len(set(read_files(test_paths[settings_string][ds])))
            for ds in datasets
        }

        initial_models = list(matches_paths.keys())
        current_models = copy.deepcopy(initial_models)

        while len(current_models) >= 1:
            combo_name = "-".join(sorted([m.replace("-", "") for m in current_models]))

            for ds in datasets:
                combined_matches = set()
                for model in current_models:
                    combined_matches.update(set(read_files(matches_paths[model][settings_string][ds])))
                match_percentage = round(len(combined_matches) / test_passwords[ds] * 100, 2)
                stats[settings_string][combo_name][ds] = [test_passwords[ds], len(combined_matches), match_percentage]

            if len(current_models) == 1:
                break

            removal_scores = {}

            for model in current_models:
                temp_models = [m for m in current_models if m != model]
                temp_score = 0
                for ds in datasets:
                    tmp_matches = set()
                    for m in temp_models:
                        tmp_matches.update(set(read_files(matches_paths[m][settings_string][ds])))
                    temp_score += len(tmp_matches)
                removal_scores[model] = temp_score

            worst_model = max(removal_scores, key=removal_scores.get)
            current_models.remove(worst_model)

    return stats


class RQ6_2(Evaluator):
    def __init__(self, test_settings, search_settings, csv_settings):
        super().__init__(test_settings, search_settings, csv_settings)

    def _get_entries(self):
        pass

    def _compute_metrics(self, matches_paths, real_paths):
        data = multi_models_attack(matches_paths, real_paths)
        for setting_string in data:
            for combo in data[setting_string]:
                for dataset in data[setting_string][combo]:
                    test_settings, n_samples = setting_string.split(os.sep)
                    stats = data[setting_string][combo][dataset]
                    variable_data = [[stats[0], stats[1], stats[2]]]
                    path, rows = self.prepare_to_csv(combo, dataset, test_settings, n_samples, variable_data)
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
        'test_name': 'rq6.2',
        'fieldnames': ["combo", "train-dataset", "test-settings", "n_samples", "test-size", "matches",
                      "match_percentage"]
    }

    test_settings['overwrite'] = True

    evaluator = RQ6_2(test_settings, search_settings, csv_settings)
    return evaluator.written_rows