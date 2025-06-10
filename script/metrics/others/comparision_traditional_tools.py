import os

from script.config.config import *
from script.test.tester import Tester

RESULTS_PATH = "results"

def _run_tester(test_settings):
    tester = Tester(test_settings)
    tester.prepare_environment()
    tester.run_test()
    csv_rows = tester.written_rows
    return csv_rows

def _prepare_settings(test, model, dataset, setting_string, display_logs):
    others, n_samples = setting_string.split(os.sep)
    char_bag, max_length, train_chunk_percentage, train_split_percentage = others.split("-")
    args_settings = \
            {'models': model,
            'max_length': int(max_length),
            'train_datasets': dataset,
            'n_samples': int(n_samples),
            'test_config': os.path.join("config", "test", f"{test}.yaml"),
            'train_chunk_percentage': int(train_chunk_percentage),
            'train_split_percentage': int(train_split_percentage),
            'char_bag': inverse_char_bag_mapping[char_bag],
            'autoload': 1,
            'overwrite': 0,
            'display_logs': display_logs,
            }

    return build_args_settings(args_settings)

def _run_missing_entry(test, model, dataset, setting_string, display_logs):
    try:
        test_settings = _prepare_settings(test, model, dataset, setting_string, display_logs)
        rows = _run_tester(test_settings)
        return rows
    except Exception as e:
        return {}

def _prepare_settings_strings(settings):
    test_settings = [
        "-".join([
            char_bag_mapping[char_bag],
            str(max_length),
            str(train_chunk),
            str(train_split)
        ])
        for char_bag in settings['char_bag']
        for max_length in settings['max_length']
        for train_chunk in settings['train_chunk_percentage']
        for train_split in settings['train_split_percentage']
    ]

    settings_strings = [os.path.join(string, str(n_samples)) for string in test_settings for n_samples in
                        settings['n_samples']]
    return settings_strings

def _compute_metrics(test_settings):
    written_rows = {}

    models = [m.replace("-", "") for m in test_settings["models"]]
    datasets = test_settings["train_datasets"]
    test = test_settings["test"]
    display_logs = test_settings.get("display_logs", 0)

    setting_strings = _prepare_settings_strings(test_settings)

    for model in models:
        for dataset in datasets:
            for setting_string in setting_strings:
                rows = _run_missing_entry(test, model, dataset, setting_string, display_logs)
                for path in rows:
                    if path not in written_rows:
                        written_rows[path] = []
                    for row in rows[path]:
                        written_rows[path].append(row)

    return written_rows

def main(test_settings):
    written_rows = _compute_metrics(test_settings)
    return written_rows