import copy
import sys
import os
import importlib.util
import inspect
import random
import itertools

from script.dataset.file_filterer import FileFilterer
from script.config.config import read_config, read_model_args
from script.test.hash import construct_hash
from script.utils.file_operations import read_datasets, save_split

PATH_TO_CONFIG = "./config/general_settings.yaml"
PATH_TO_MODEL_CONFIG = "./config/model/model_settings.yaml"
PREPROCESSING_FUNCTIONS_PATH = "./script/dataset/preprocessing/"
sys.path.append(PREPROCESSING_FUNCTIONS_PATH)

CHAR_BAG_CHARS_NUMBERS_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `"
CHAR_BAG_CHARS_NUMBERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_BAG_CHARS_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `"
CHAR_BAG_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_BAG_NUMBERS = "0123456789"

char_bag_mapping = {
    CHAR_BAG_CHARS_NUMBERS_SYMBOLS: "all",
    CHAR_BAG_CHARS_NUMBERS: "c+n",
    CHAR_BAG_CHARS_SYMBOLS: "c+s",
    CHAR_BAG_CHARS: "c",
    CHAR_BAG_NUMBERS: "n"
}


class Tester:
    def __init__(self, args):
        self.file_filterer = FileFilterer()

        func_dict = self.extract_functions(PREPROCESSING_FUNCTIONS_PATH)

        settings = read_config(PATH_TO_CONFIG)
        if "train_datasets" not in settings['pre_split_params']:
            settings['split_params']['train_datasets'] = self.file_filterer.get_datasets_from_default_settings()

        args_settings = self.read_args_settings(args)

        if "test_config" in args_settings["general_params"]:
            test_config = args_settings["general_params"]["test_config"]
        else:
            test_config = settings["general_params"]["test_config"]

        test_settings = {}
        for test in test_config:
            test_settings.update(read_config(test))

        self.settings = self.update_settings(settings, args_settings, test_settings)
        self.dict_param_to_type = self.map_param_to_type(self.settings)
        self.func_dict = self.get_preprocess_func(func_dict, test_settings)

    def extract_functions(self, path):
        func_dict = {}
        for file_name in os.listdir(path):
            full_path = os.path.join(path, file_name)
            if not full_path.endswith(".py"):
                continue

            file_name = file_name.replace(".py", "")
            func_dict[file_name] = {}

            spec = importlib.util.spec_from_file_location(file_name, full_path)
            module = importlib.util.module_from_spec(spec)

            spec.loader.exec_module(module)

            for func_name, func_handle in inspect.getmembers(module, inspect.isfunction):
                func_dict[file_name][func_name] = func_handle

        return func_dict

    def read_args_settings(self, args):
        args_settings = {}
        for type in args:
            args_settings[type] = {}
            for key in args[type]:
                val = args[type][key]
                if val is not None:
                    args_settings[type][key] = val
        return args_settings

    def update_settings(self, settings, args_settings, test_settings):
        final_settings = {}

        for test_name, test_dict in test_settings.items():
            final_settings[test_name] = {}
            for params_type in ['general_params', 'pre_split_params', 'split_params', 'post_split_params',
                                'test_params']:
                final_settings[test_name][params_type] = {}

                def update_values(source_dict, params_type):
                    if params_type in list(source_dict.keys()) and source_dict[params_type] is not None:
                        for option in source_dict[params_type]:
                            if isinstance(source_dict[params_type][option], dict):
                                for key in source_dict[params_type][option]:
                                    value = source_dict[params_type][option][key]
                                    if isinstance(value, int):
                                        value = [value]
                                    if key not in final_settings[test_name][params_type] and option != "test_config":
                                        final_settings[test_name][params_type][key] = value
                            else:
                                value = source_dict[params_type][option]
                                if isinstance(value, int):
                                    value = [value]
                                if option not in final_settings[test_name][params_type] and option != "test_config":
                                    final_settings[test_name][params_type][option] = value

                update_values(args_settings, params_type)
                update_values(test_dict, params_type)
                update_values(settings, params_type)

        return final_settings

    def get_preprocess_func(self, func_dict, test_settings):
        function_and_handles = {}
        for test_name, test_dict in test_settings.items():
            function_and_handles[test_name] = {}
            for params_type in test_dict:
                for step in test_dict[params_type]:
                    key = step.split(".")
                    values = list(test_dict[params_type][step].keys())
                    file_name = key[0]
                    function_name = key[1]

                    if function_name in func_dict[file_name]:
                        function_and_handles[test_name][function_name] = [func_dict[file_name][function_name], values]
                    else:
                        error_string = f"Function {function_name} not found in {file_name}.py"
                        raise ModuleNotFoundError(error_string)

        return function_and_handles

    def map_param_to_type(self, settings):
        dict_param_to_type = {}
        for test_name, test_dict in settings.items():
            for params_type in test_dict:
                for option in test_dict[params_type]:
                    dict_param_to_type[option] = params_type
        return dict_param_to_type

    def generate_combinations(self, test_args):
        keys = []
        values = []
        for params_type in test_args:
            for key, value in test_args[params_type].items():
                keys.append(key)
                values.append(value)

        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def get_train_test_datasets_path(self, test_settings):
        train_dataset = test_settings["train_datasets"]
        path_train_datasets = sorted(self.file_filterer.select_from_names(train_dataset))

        path_test_datasets = []
        if 'test_datasets' in test_settings:  # if cross dataset
            test_dataset = test_settings["test_datasets"]
            path_test_datasets = sorted(self.file_filterer.select_from_names(test_dataset))

        return path_train_datasets, path_test_datasets

    def run_test(self):
        for name, values in self.settings.items():
            test_name = name
            test_args = values

            data_to_embed = None
            if "data_to_embed" in test_args["general_params"]:
                data_to_embed = test_args["general_params"]["data_to_embed"]
                test_args["general_params"].pop("data_to_embed")

            combinations = self.generate_combinations(test_args)

            for combination in combinations:
                if data_to_embed is not None:
                    combination["data_to_embed"] = data_to_embed
                    assert combination["models"] == "passflow", "The embedding must be done with passflow's encoder."

                assert not (("autoload" in combination) and ("path_to_checkpoint" in combination)), \
                    "You can not pass both autoload and path_to_checkpoint!"

                train_hash = construct_hash(combination, self.dict_param_to_type, "train")
                test_hash = construct_hash(combination, self.dict_param_to_type, "test")

                self.run_specific_test(combination, test_name, train_hash, test_hash)

    def run_specific_test(self, test_settings, test_name, train_hash, test_hash):
        random.seed(42)  # setting seed for reproducibility

        train_data_path = os.path.join(self.file_filterer.train_and_test_path, "train-" + str(train_hash) + ".pickle")
        test_data_path = os.path.join(self.file_filterer.train_and_test_path, "test-" + str(test_hash) + ".pickle")

        skip = os.path.exists(train_data_path) and os.path.exists(test_data_path)

        if not skip:
            path_train_datasets, path_test_datasets = self.get_train_test_datasets_path(test_settings)
            train_passwords = read_datasets(path_train_datasets)
            test_passwords = read_datasets(path_test_datasets)  # test_passwords = [] if not cross_dataset

            for function_name in self.func_dict[test_name]:
                function, args = self.func_dict[test_name][function_name]
                kwargs_dict = {key: test_settings[key] for key in args if
                               key not in ["train_datasets", "test_datasets"]}
                train_passwords, test_passwords = function(train_passwords, test_passwords, **kwargs_dict)

            test_passwords = set(test_passwords)

            save_split(train_passwords, train_data_path)
            save_split(test_passwords, test_data_path)

        print(f"Running test {test_name}")
        self.run_models(test_settings, test_name, train_hash, train_data_path, test_hash, test_data_path)

    def import_model(self, path, class_name):
        module = importlib.import_module(path)
        model_class = getattr(module, class_name)
        return model_class

    def run_models(self, test_settings, test_name, train_hash, train_data_path, test_hash, test_data_path):
        models_settings = read_config(PATH_TO_MODEL_CONFIG)

        model_name = test_settings["models"]
        path_to_class, class_name, path_to_config, test_args = read_model_args(models_settings, str(model_name))

        model_class = self.import_model(path_to_class, class_name)

        output_path = self.construct_output_path(test_settings, test_name, model_name)

        settings = {'train_path': str(train_data_path),
                    'train_hash': str(train_hash),
                    'test_path': str(test_data_path),
                    'test_hash': str(test_hash),
                    'test_args': test_args,
                    'config_file': str(path_to_config),
                    'max_length': str(test_settings["max_length"]),
                    'n_samples': str(test_settings["n_samples"]),
                    'output_path': str(output_path),
                    'path_to_checkpoint': str(test_settings["path_to_checkpoint"]) if "path_to_checkpoint" in
                                                                                      test_settings else False,
                    'autoload': str(test_settings["autoload"]) if "autoload" in test_settings else False,
                    'display_logs': str(test_settings["display_logs"]) if "display_logs" in test_settings else False,
                    'data_to_embed': test_settings["data_to_embed"] if "data_to_embed" in test_settings else False,
                    }

        if 'test_reference' in test_settings:
            if test_settings["test_reference"].endswith(".yaml"):
                reference_path = self.construct_reference_path(test_settings, train_hash, model_name)
                settings["guesses_dir"] = reference_path
            else:
                settings["guesses_dir"] = test_settings["test_reference"]

        if 'use_existing_samples' in test_settings:
            use_existing_samples = test_settings['use_existing_samples']

            tmp_settings = copy.deepcopy(test_settings)
            tmp_settings['n_samples'] = use_existing_samples

            use_existing_samples_path = self.construct_output_path(tmp_settings, test_name, model_name)
            use_existing_samples_path = os.path.join(use_existing_samples_path, test_hash, "guesses")

            settings["use_existing_samples_path"] = use_existing_samples_path

        for setting in test_args:
            settings[setting] = test_args[setting]

        print(f"Starting {model_name}:")
        model_class(settings)

    def construct_output_path(self, test_settings, test_name, model_name):
        result_list = []

        train_dataset = test_settings["train_datasets"]
        n_samples = str(test_settings["n_samples"])

        skip_list = ["autoload", "display_logs", "models", "test_reference", "train_datasets", "n_samples",
                     "use_existing_samples", "path_to_checkpoint"]

        for key in sorted(test_settings.keys()):
            value = test_settings[key]
            if key in skip_list:
                continue

            if key == "char_bag":
                value = char_bag_mapping.get(test_settings[key], test_settings[key])

            result_list.append(''.join(str(value)))

        test_args = '-'.join(result_list)

        output_path = os.path.join("results", test_name, model_name, train_dataset, test_args, n_samples)
        return output_path

    def construct_reference_path(self, test_settings, train_hash, model_name):
        test_reference = test_settings["test_reference"]

        test_reference_name = os.path.basename(test_reference)
        ext_idx = test_reference_name.rfind(".")
        test_reference_name = test_reference_name[:ext_idx]

        reference_test_settings = next(iter(read_config(test_reference).values()))
        reference_test_parameters = self.get_parameters(reference_test_settings)

        keep_list = ['train_datasets', 'models', 'n_samples', 'test_datasets',
                     'train_split_percentage', 'train_chunk_percentage']

        for setting in copy.deepcopy(test_settings):
            if (setting not in reference_test_parameters) and (setting not in keep_list):
                del test_settings[setting]

        for mode_setting in reference_test_settings:
            for function in reference_test_settings[mode_setting]:
                for setting in reference_test_settings[mode_setting][function]:
                    test_settings[setting] = reference_test_settings[mode_setting][function][setting][0]

        reference_path = self.construct_output_path(test_settings, test_reference_name, model_name)
        reference_path = os.path.join(reference_path, train_hash, "guesses")
        return reference_path

    def get_parameters(self, settings):
        parameters = []
        for key, value in settings.items():
            if isinstance(value, dict):
                parameters.extend(self.get_parameters(value))
            else:
                parameters.append(key)
        return parameters
