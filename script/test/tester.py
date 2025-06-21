import sys
import csv
import os
import importlib.util
import inspect
import random
import itertools

from script.utils.download_raw_data import main as download_raw_data

from script.dataset.file_filterer import FileFilterer
from script.config.config import *
from script.test.hash import construct_hash
from script.utils.file_operations import save_split, reset_stdout, reset_stderr
from script.utils.preprocessing_utils import SkipCombinationException

PREPROCESSING_FUNCTIONS_PATH = "./script/dataset/preprocessing/"
DOWNLOAD_DATASETS_SCRIPT = "script/utils/download_raw_data.py"
sys.path.append(PREPROCESSING_FUNCTIONS_PATH)


class Tester:
    def __init__(self, args):
        self.args_settings = read_args_settings(args)

        test_config = self._get_test_config_path()

        self.test_settings = read_config(test_config)
        self.evaluation_script = self.test_settings[next(iter(self.test_settings))].pop("evaluation_script", None)
        self.figure_script = self.test_settings[next(iter(self.test_settings))].pop("figure_script", None)
        self.written_rows = None

    def _get_test_config_path(self):
        test_config = self.args_settings["general_params"].get("test_config")

        if not test_config:
            raise ValueError("No test config file was provided.")

        return test_config

    def _extract_script_path(self):
        test_name = next(iter(self.test_settings))
        return self.test_settings[test_name].pop("script", None)

    def _extract_preprocessing_functions(self, path):
        func_dict = {}

        for file in os.listdir(path):
            if not file.endswith(".py"):
                continue

            module_name = file[:-3]
            file_path = os.path.join(path, file)

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            func_dict[module_name] = {
                name: func for name, func in inspect.getmembers(module, inspect.isfunction)
            }

        return func_dict

    def _get_preprocessing_functions(self, func_dict):
        function_map = {}

        for test_name, test_dict in self.test_settings.items():
            function_map[test_name] = {}

            for param_type, steps in test_dict.items():
                if param_type == "general_params":
                    continue

                for step in steps:
                    file_name, func_name = step.split(".")
                    keys = list(steps[step].keys())

                    if func_name in func_dict.get(file_name, {}):
                        function_map[test_name][func_name] = [func_dict[file_name][func_name], keys]
                    else:
                        raise ModuleNotFoundError(f"Function {func_name} not found in {file_name}.py")

        return function_map

    def _load_preprocessing_functions(self):
        all_functions = self._extract_preprocessing_functions(PREPROCESSING_FUNCTIONS_PATH)
        return self._get_preprocessing_functions(all_functions)

    def _prepare_script_input(self):
        arg_tmp = get_keys_and_values(self.args_settings)
        final_settings = get_keys_and_values(self.test_settings)

        for setting in final_settings:
            if setting in arg_tmp:
                final_settings[setting] = arg_tmp[setting]
        return final_settings

    def _import_script(self, path):
        module_path = os.path.abspath(path)
        module_name = os.path.splitext(os.path.basename(module_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "main"):
            raise AttributeError(f"The module {module_path} does not contain a 'main' function.")

        return module

    def prepare_environment(self):
        self.file_filterer = FileFilterer()
        self.settings = update_settings(self.args_settings, self.test_settings)
        self.dict_param_to_type = map_param_to_type(self.settings)
        self.func_dict = self._load_preprocessing_functions()
        self.written_rows = {}

    def prepare_script_settings(self):
        self.settings = self._prepare_script_input()

    def execute_eval_script(self):
        module = self._import_script(self.evaluation_script)
        self.written_rows = module.main(self.settings)

    def execute_figure_script(self, csv_rows=None):
        module = self._import_script(self.figure_script)
        settings = self.settings[next(iter(self.settings))]

        if not isinstance(settings,dict):  # If it's an evaluation script, settings are already formatted
            settings = self.settings
        else:
            settings = get_keys_and_values(settings)
        module.main(csv_rows, settings)

    def generate_combinations(self, test_args):
        keys, values = [], []
        n_samples = 0
        for params_type in test_args:
            for key, value in test_args[params_type].items():
                if key == "n_samples":
                    n_samples = value
                    continue

                keys.append(key)
                values.append(value)

        combinations = []
        for combo in itertools.product(*values):
            combo_dict = dict(zip(keys, combo))
            combo_dict['n_samples'] = n_samples
            if not ('test_datasets' in combo_dict and combo_dict['train_datasets'] == combo_dict['test_datasets']):
                combinations.append(combo_dict)

        return combinations

    def download_dataset(self, file_path, name):
        if not os.path.isfile(file_path):
            print(f"[INFO] Dataset not found at '{file_path}'. Attempting automatic download...")
            download_raw_data(chosen_datasets=[name], datasets_folder=self.file_filterer.datasets_path)

    def get_train_test_datasets_path(self, test_settings):
        train_dataset = test_settings["train_datasets"]

        path_train_datasets = self.file_filterer.select_from_names(train_dataset)
        self.download_dataset(path_train_datasets, train_dataset)

        if 'test_datasets' in test_settings:  # if cross dataset
            test_dataset = test_settings["test_datasets"]
            path_test_datasets = self.file_filterer.select_from_names(test_dataset)
            self.download_dataset(path_test_datasets, test_dataset)
            return [path_train_datasets], [path_test_datasets]

        return [path_train_datasets], []

    def run_test(self):
        for name, values in self.settings.items():
            test_name = name
            test_args = values

            data_to_embed = None
            if "data_to_embed" in test_args["general_params"]:
                data_to_embed = test_args["general_params"]["data_to_embed"]
                test_args["general_params"].pop("data_to_embed")

            combinations = self.generate_combinations(test_args)

            skipped_thresholds = {}

            for combination in combinations:
                if data_to_embed is not None:
                    combination["data_to_embed"] = data_to_embed
                    assert combination["models"] == "passflow", "The embedding must be done with passflow's encoder."

                assert not (("autoload" in combination) and ("path_to_checkpoint" in combination)), \
                    "You can not pass both autoload and path_to_checkpoint!"

                skip_key = tuple((k, make_hashable(v)) for k, v in combination.items() if k != "train_chunk_percentage" and k != "models")
                if skip_key in skipped_thresholds:
                    if combination.get("train_chunk_percentage", 0) >= skipped_thresholds[skip_key]:
                        print(f"[SKIP] Skipping combination {combination} because train_chunk_percentage is too high.")
                        continue

                train_hash = construct_hash(combination, self.dict_param_to_type, "train")
                test_hash = construct_hash(combination, self.dict_param_to_type, "test")

                try:
                    self.run_specific_test(combination, test_name, train_hash, test_hash)

                except SkipCombinationException as e:
                    print(f"[INFO] {e} Skipping combination: {combination}")
                    if "train_chunk_percentage is larger than dataset size." in str(e):
                        current = combination["train_chunk_percentage"]
                        if skip_key not in skipped_thresholds or current < skipped_thresholds[skip_key]:
                            skipped_thresholds[skip_key] = current

    def custom_key_order(self, d):
        keys = list(d.keys())

        keys = [k for k in keys if k not in ('read_train_passwords', 'read_test_passwords')]

        ordered_keys = []
        if 'read_train_passwords' in d:
            ordered_keys.append('read_train_passwords')
        if 'read_test_passwords' in d:
            ordered_keys.append('read_test_passwords')
        ordered_keys += keys

        return {k: d[k] for k in ordered_keys}

    def run_specific_test(self, test_settings, test_name, train_hash, test_hash):
        random.seed(42)  # setting seed for reproducibility

        output_path = self.construct_output_path(test_settings, test_name, test_settings["models"])

        skip_gen = False
        overwrite = test_settings.get("overwrite", False)
        if not overwrite:
            missing_n_samples, found_rows = self.get_row_from_previous_runs(test_settings, test_hash, output_path)

            if found_rows:
                for path in found_rows:
                    if path not in self.written_rows:
                        self.written_rows[path] = []
                    for row in found_rows[path]:
                        self.written_rows[path].append(row)

            if not missing_n_samples:
                skip_gen = True
            else:
                test_settings['n_samples'] = missing_n_samples
                new_max_n_samples = max(missing_n_samples)
                parts = output_path.split(os.sep)
                parts[-1] = str(new_max_n_samples)
                output_path = os.sep.join(parts)

        if not skip_gen:
            train_data_path = os.path.join(self.file_filterer.train_and_test_path, "train-" + str(train_hash) + ".pickle")
            test_data_path = os.path.join(self.file_filterer.train_and_test_path, "test-" + str(test_hash) + ".pickle")

            skip = os.path.exists(train_data_path) and os.path.exists(test_data_path)

            if not skip:
                path_train_datasets, path_test_datasets = self.get_train_test_datasets_path(test_settings)

                train_passwords = []
                test_passwords = []

                ordered_keys = self.custom_key_order(self.func_dict[test_name])

                for function_name in ordered_keys:
                    function, args = self.func_dict[test_name][function_name]

                    kwargs_dict = {
                        key: (
                            path_train_datasets if key == "train_datasets"
                            else path_test_datasets if key == "test_datasets"
                            else test_settings[key]
                        )
                        for key in args
                    }

                    train_passwords, test_passwords = function(train_passwords, test_passwords, **kwargs_dict)

                test_passwords = set(test_passwords)

                save_split(train_passwords, train_data_path)
                save_split(test_passwords, test_data_path)

            print(f"Running test {test_name}")
            if test_settings["models"] != "NULL":
                self.run_models(test_settings, output_path, train_hash, train_data_path, test_hash, test_data_path)

    def import_model(self, path, class_name):
        module = importlib.import_module(path)
        model_class = getattr(module, class_name)
        return model_class

    def run_models(self, test_settings, output_path, train_hash, train_data_path, test_hash, test_data_path):
        models_settings = read_config(PATH_TO_MODEL_CONFIG)

        model_name = test_settings["models"]
        path_to_class, class_name, path_to_config, sampling_args = read_model_args(models_settings, str(model_name))

        model_class = self.import_model(path_to_class, class_name)

        settings = {'model_name': str(model_name),
                    'train_path': str(train_data_path),
                    'train_hash': str(train_hash),
                    'test_path': str(test_data_path),
                    'test_hash': str(test_hash),
                    'config_file': str(path_to_config),
                    'max_length': str(test_settings["max_length"]),
                    'n_samples': test_settings["n_samples"],
                    'output_path': str(output_path),
                    'path_to_checkpoint': test_settings.get("path_to_checkpoint", False),
                    'autoload': test_settings.get("autoload", False),
                    'display_logs': test_settings.get("display_logs", False),
                    'data_to_embed': test_settings.get("data_to_embed", False),
                    'guesses_file': test_settings.get("guesses_file", False),
                    'overwrite': test_settings.get("overwrite", False),
                    'sub_samples_from_file': test_settings.get("sub_samples_from_file", False),
                    'save_guesses': test_settings.get("save_guesses", False),
                    'save_matches': test_settings.get("save_matches", False),
                    }

        for setting in sampling_args:
            settings[setting] = sampling_args[setting]

        print(f"Starting {model_name}:")
        model = model_class(settings)
        rows = model.written_rows

        for path in rows:
            if path not in self.written_rows:
                self.written_rows[path] = []
            for row in rows[path]:
                self.written_rows[path].append(row)

        reset_stdout()
        reset_stderr()

    def construct_output_path(self, test_settings, test_name, model_name):
        result_list = []

        train_dataset = test_settings["train_datasets"]
        n_samples = test_settings["n_samples"]
        if isinstance(n_samples, list):
            n_samples = str(max(n_samples))
        else:
            n_samples = str(n_samples)

        for key in sorted(test_settings.keys()):
            value = test_settings[key]
            if key == 'train_datasets' or self.dict_param_to_type[key] == 'general_params':
                continue

            if key == "char_bag":
                value = char_bag_mapping.get(test_settings[key], test_settings[key])

            result_list.append(''.join(str(value)))

        test_args = '-'.join(result_list)

        output_path = os.path.join("results", test_name, model_name, train_dataset, test_args, n_samples)
        return output_path

    def check_from_csv(self, csv_path, query):
        last_match = None
        try:
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if all(row.get(k) == v or row.get(k) == 'NONE' for k, v in query.items()):
                        last_match = ",".join(row[k] for k in row)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {csv_path}")
            return None

        return last_match

    def get_row_from_previous_runs(self, test_settings, test_hash, output_path):
        infos = output_path.split(os.sep)
        csv_path = os.path.join(infos[0], infos[1], f"{infos[1]}.csv")

        missing_queries = []
        found_rows = {csv_path: []}

        for n_samples in test_settings["n_samples"]:
            missing_queries.append({
                'model': test_settings["models"],
                'train-dataset': test_settings["train_datasets"],
                'test-settings': infos[4],
                'test-hash': test_hash,
                'n_samples': str(n_samples),
            })

        for query in missing_queries[::]:
            row = self.check_from_csv(csv_path, query)
            if row:
                found_rows[csv_path].append(row)
                missing_queries.remove(query)

        missing_n_samples = []
        for query in missing_queries:
            missing_n_samples.append(int(query['n_samples']))

        return missing_n_samples, found_rows

def make_hashable(v):
    if isinstance(v, list):
        return tuple(v)
    elif isinstance(v, dict):
        return tuple(sorted((k, make_hashable(vv)) for k, vv in v.items()))
    return v