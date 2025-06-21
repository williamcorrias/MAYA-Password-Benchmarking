import os
import re
import pickle
import gzip
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

def _compute_pattern_distribution(path):
    distribution = {}
    for pattern in regex:
        distribution[pattern] = 0

    total_passwords = 0

    compiled_regex = {k: re.compile(v) for k, v in regex.items()}

    for chunk in read_chunk(path):
        for password in chunk:
            if not password:
                continue

            password = password.rstrip()
            total_passwords += 1

            for id, pattern in compiled_regex.items():
                if pattern.fullmatch(password):
                    distribution[id] += 1

    stats = {}
    for pattern in distribution:
        stats[pattern] = [distribution[pattern], round(distribution[pattern] / total_passwords * 100, 2)]

    return stats

class RQ7_3Evaluator(Evaluator):
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

                    stats = _compute_pattern_distribution(file_path)

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
        'test_name': 'rq7.3',
        'fieldnames' : ["model", "train-dataset", "test-settings", "n_samples", "pattern", "matches", "match_percentage"]
    }

    evaluator = RQ7_3Evaluator(test_settings, search_settings, csv_settings)
    return evaluator.written_rows