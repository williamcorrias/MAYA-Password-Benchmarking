import os

import pickle
import gzip
from script.metrics.statistics.evaluator import Evaluator

def read_chunk(file, chunk_size=20480):
    def read_lines(file_obj):
        chunk = []
        for line in file_obj:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    if file.endswith('.gz'):
        with gzip.open(file, 'rt') as f:
            yield from read_lines(f)
    elif file.endswith('.txt'):
        with open(file, 'r') as f:
            yield from read_lines(f)
    elif file.endswith('.pickle'):
        with open(file, 'rb') as f:
            yield pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file}")

def _compute_length_distribution(path):
    distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    total_passwords = 0
    for chunk in read_chunk(path):
        for password in chunk:
            if not password:
                continue

            password = password.rstrip()
            length = len(password)

            if length in distribution:
                distribution[length] += 1
                total_passwords += 1

    stats = {}
    for length in distribution:
        stats[length] = [distribution[length], round(distribution[length] / total_passwords * 100, 2)]

    return stats

class RQ7_2Evaluator(Evaluator):
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

    def _compute_metrics(self, guesses_paths, real_paths):
        guesses_paths['real'] = real_paths

        for model in guesses_paths:
            for setting_string in guesses_paths[model]:
                for dataset in guesses_paths[model][setting_string]:

                    file_path = guesses_paths[model][setting_string][dataset]

                    stats = _compute_length_distribution(file_path)

                    test_settings, n_samples = setting_string.split(os.sep)
                    variable_data = []
                    for key in stats:
                        variable_data.append([key, stats[key][0], stats[key][1]])

                    path, rows = self.prepare_to_csv(model, dataset, test_settings, n_samples, variable_data)

                    if path not in self.written_rows:
                        self.written_rows[path] = []
                    for row in rows:
                        self.written_rows[path].append(row)

def main(test_settings):
    search_settings = {
        'mode': "guesses",
        'real_data_mode': "full",
    }

    csv_settings = {
        'test_name': 'rq7.2',
        'fieldnames' : ["model", "train-dataset", "test-settings", "n_samples", "length", "matches", "match_percentage"]
    }

    evaluator = RQ7_2Evaluator(test_settings, search_settings, csv_settings)
    return evaluator.written_rows